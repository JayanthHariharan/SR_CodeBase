from AlgorithmImports import *
from datetime import timedelta, datetime
from math import isfinite

class BasicMeanReversionAlgorithm(QCAlgorithm):

    def Initialize(self):
        # Dates & capital
        self.SetStartDate(2019, 1, 1)
        self.SetEndDate(2025, 1, 1)
        self.SetCash(100_000)

        # Keep a tiny cash buffer to avoid BP rejections
        self.Settings.FreePortfolioValuePercentage = 0.01  # 1% idle cash

        # Asset & frictions
        equity = self.AddEquity("SPY", Resolution.Daily)
        equity.SetDataNormalizationMode(DataNormalizationMode.Raw)  # flip to Adjusted when factor files are current
        equity.SetLeverage(1.0)
        from QuantConnect.Orders.Fees import ConstantFeeModel
        from QuantConnect.Orders.Slippage import ConstantSlippageModel
        equity.SetFeeModel(ConstantFeeModel(0.0))
        equity.SetSlippageModel(ConstantSlippageModel(0.0001))  # ~1bp
        self.symbol = equity.Symbol

        # --- Core MR params (optimizer can override) ---
        self.window     = int(self.GetParameter("window", 20))
        self.entry      = float(self.GetParameter("entry", 2.0))      # z to ENTER long
        self.exit       = float(self.GetParameter("exit", 1.0))       # |z| to EXIT
        self.atr_mult   = float(self.GetParameter("atr_mult", 2.5))   # ATR stop multiple
        self.vol_target = float(self.GetParameter("vol_target", 0.01))# target daily vol (1%)
        self.cooldown   = int(self.GetParameter("cooldown", 3))       # days after exit
        self.reglen     = int(self.GetParameter("reglen", 200))       # regime SMA length
        self.time_stop  = int(self.GetParameter("time_stop", 15))     # max bars in trade

        # --- Sentiment knobs (OFF by default; safe no-ops) ---
        self.use_sentiment = int(self.GetParameter("use_sentiment", 0))
        self.sent_gate     = float(self.GetParameter("sent_gate", -0.10))
        self.sent_entry_k  = float(self.GetParameter("sent_entry_k", 0.25))
        self.sent_size_k   = float(self.GetParameter("sent_size_k", 0.25))

        # Indicators
        self.price  = self.SMA(self.symbol, 1)
        self.sma    = self.SMA(self.symbol, self.window)
        self.std    = self.STD(self.symbol, self.window)
        self.regime = self.SMA(self.symbol, self.reglen)
        self.atr    = self.ATR(self.symbol, 14, MovingAverageType.Wilders)
        self.SetWarmUp(max(self.reglen, self.window, 20), Resolution.Daily)

        # Order state
        self.entryTicket = None   # type: OrderTicket | None
        self.stopTicket  = None   # type: OrderTicket | None
        self.entryBarTime = None  # type: datetime | None
        self.lastExitTime = datetime(1900, 1, 1)

        # Status set that means "done" (don't cancel)
        self.SAFE_DONE = {OrderStatus.Filled, OrderStatus.Canceled, OrderStatus.Invalid}

    # ====== MAIN ======
    def OnData(self, data: Slice):
        bar = data.Bars.get(self.symbol)
        if not bar or self.IsWarmingUp:
            return

        px = float(bar.Close)
        mu = self.sma.Current.Value
        sd = max(self.std.Current.Value, 1e-12)
        z  = (px - mu) / sd

        invested = self.Portfolio[self.symbol].Invested
        in_cooldown = (self.Time - self.lastExitTime) <= timedelta(days=self.cooldown)
        bullish = px > self.regime.Current.Value

        # Sentiment controls (OFF → s=0)
        s = 0.0
        gate_ok = (s >= self.sent_gate) if self.use_sentiment else True
        entry_eff = self.entry
        if self.use_sentiment:
            entry_eff *= (1.0 - self.sent_entry_k * s)
            entry_eff = min(max(0.5*self.entry, entry_eff), 2.0*self.entry)

        # ---- ENTRY: long-only, regime & cooldown respected ----
        if (not invested) and bullish and gate_ok and (z <= -entry_eff) and (not in_cooldown):
            # Volatility-targeted base weight
            realized_vol = self.std.Current.Value / max(mu, 1e-12)
            w = self.vol_target / max(realized_vol, 1e-6)
            w = max(0.0, min(1.0, w))
            # Scale with signal strength
            w *= min(1.5, max(0.5, abs(z)/max(entry_eff, 1e-6)))
            # Sentiment scaling
            if self.use_sentiment:
                w *= (1.0 + self.sent_size_k * s)
            w = max(0.0, min(1.0, w))

            delta = self._delta_shares_for_target(w)
            if delta > 0:
                if self.entryTicket and self.entryTicket.Status not in self.SAFE_DONE:
                    self.entryTicket.Cancel()
                self.entryTicket = self.MarketOnOpenOrder(self.symbol, delta)
                self.entryBarTime = self.Time

        # ---- EXIT: mean reversion achieved OR time stop ----
        elif invested:
            bars_in_trade = (self.Time - (self.entryBarTime or self.Time)).days
            if abs(z) <= self.exit or bars_in_trade >= self.time_stop:
                self._exit_and_cancel_stop()

        # Plots
        self.Plot("Data", "Close", px)
        self.Plot("Signals", "z", z)
        self.Plot("Signals", "0", 0)

    def OnOrderEvent(self, oe: OrderEvent):
        # When long fills, (re)place ATR stop
        if self.entryTicket and oe.OrderId == self.entryTicket.OrderId and oe.Status == OrderStatus.Filled:
            qty = self.Portfolio[self.symbol].Quantity
            if qty > 0:
                px = oe.FillPrice
                stop = px - self.atr_mult * self.atr.Current.Value
                if self.stopTicket and self.stopTicket.Status not in self.SAFE_DONE:
                    self.stopTicket.Cancel()
                self.stopTicket = self.StopMarketOrder(self.symbol, -qty, stop)

        # Track exits (for cooldown)
        if oe.Status == OrderStatus.Filled and oe.FillQuantity < 0 and self.Portfolio[self.symbol].Quantity == 0:
            self.lastExitTime = self.Time
            self.entryBarTime = None

    # ====== HELPERS ======
    def _exit_and_cancel_stop(self):
        if self.stopTicket and self.stopTicket.Status not in self.SAFE_DONE:
            self.stopTicket.Cancel()
        self.stopTicket = None
        self.Liquidate(self.symbol)
        self.lastExitTime = self.Time
        self.entryBarTime = None

    def _delta_shares_for_target(self, target_w: float) -> int:
        """Safe delta shares to reach target weight, respecting buying power and buffers."""
        # Cap near 100% to respect FreePortfolioValuePercentage and add tiny cushion
        cap = 1.0 - self.Settings.FreePortfolioValuePercentage - 0.002
        target_w = max(0.0, min(cap, float(target_w)))

        # Ask LEAN for the safe delta quantity (considers leverage/fees/BP)
        qty = self.CalculateOrderQuantity(self.symbol, target_w)

        # Round toward zero to lot size
        lot = self.Securities[self.symbol].SymbolProperties.LotSize
        if qty > 0:
            qty = int(qty // lot * lot)
        else:
            qty = -int((-qty) // lot * lot)

        # Extra cushion when targeting very high weights
        if qty > 0 and target_w > 0.95:
            qty = max(0, qty - lot)
        return int(qty)