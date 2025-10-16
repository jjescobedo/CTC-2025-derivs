"""
Example trading strategy for CTC Derivatives Trading Game.

Participants should implement their strategy by subclassing AbstractTradingStrategy
and implementing the required methods.

This example shows a basic strategy that calculates fair values using expected
dice roll outcomes and quotes around those fair values.

KEY CONCEPTS:

DICE MECHANICS:
- Game uses 10,000-sided dice (values 1-10,000)
- Each round gets 2000 training rolls + up to 2000 more rolls during gameplay

PRODUCTS:
- Futures ("S,F,N"): Settle to the sum die rolls in first N subrounds, linear payoff
- Call Options ("S,C,STRIKE,EXPIRY"): Pay max(0, sum - strike)
- Put Options ("S,P,STRIKE,EXPIRY"): Pay max(0, strike - sum)

TRADING:
- Called each sub-round to provide bid/ask quotes
- Return dict mapping product_id -> (bid_price, ask_price)
- Only quote on products you want to trade
- Trades execute at mid-point of crossing quotes

AVAILABLE OBJECTS IN make_market():
- marketplace: Use marketplace.get_products() to get list of tradeable products
- training_rolls: List of 2000 historical dice rolls for analysis
- current_rolls: List of dice rolls in current round (grows each sub-round)
- my_trades: Your trading history - use my_trades.get_position(product_id) to get positions
- round_info: Current round state - use round_info.get("current_sub_round", 0) for timing

DEBUGGING:
- Use print statements in on_round_end() to track performance
- Test strategies on specific sub-rounds by adding conditions
"""
from autograder.sdk.strategy_interface import AbstractTradingStrategy
import numpy as np
from typing import Any, Dict, Tuple
from scipy.stats import norm
## can also import other standard libraries as needed

class MyTradingStrategy(AbstractTradingStrategy):
    
    def __init__(self):
        self.spread_width = 4.0  # Bid-ask spread width
        self.dice_sides = 6      # Will be updated in on_game_start
        
    def on_game_start(self, config: Dict[str, Any]) -> None:
        """Initialize strategy with game configuration."""
        self.dice_sides = config.get("dice_sides", 6)
        self.team_name = config.get("team_name", "Unknown")
        print(f"Strategy {self.team_name} starting with {self.dice_sides}-sided dice")
        
    def make_market(self, *, marketplace: Any, training_rolls: Any, my_trades: Any, 
                   current_rolls: Any, round_info: Any) -> Dict[str, Tuple[float, float]]:
        """
        Parameters:
        - marketplace: Use marketplace.get_products() to access all tradeable products.
        - training_rolls: 2000 historical dice rolls for distribution analysis.
        - current_rolls: Current round dice rolls, which grows each sub-round.
        - my_trades: Your trading history.
            - my_trades.get_position(product_id) returns a Position object with:
                * .position: Your net position (positive for long, negative for short).
                * .trades: A list of your trades for that product.
                * .average_price: The volume-weighted average price of your position.
            - my_trades.get_summary() returns a dictionary with overall trading statistics.
        - round_info: A dictionary with the current round state.
            - round_info.get("current_sub_round", 0) gives the current sub-round number.
            - round_info.get("total_sub_rounds", 10) gives the total sub-rounds in the round.
        
        Returns dict mapping product_id -> (bid_price, ask_price)
        """
        # Calculate expected value per dice roll from historical data

        current_sub_round = round_info.get("current_sub_round", 0)

        mean_roll, std_dev_roll = self._get_roll_distribution_params(
            training_rolls, current_rolls, current_sub_round
        )

        if current_sub_round == 10:
            expected_total_sum = np.sum(current_rolls)
            std_dev_total_sum = 0.0
            future_half_spread = 0.5

        else:
            expected_total_sum = mean_roll * 20_000
            rolls_remaining = 20_000 - (current_sub_round * 2_000)
            std_dev_total_sum = np.sqrt(rolls_remaining) * std_dev_roll

            z_score = 2.5
            future_half_spread = max(1.0, z_score * std_dev_total_sum)
            
        quotes = dict()

        for product in marketplace.get_products():

            fair_value = self._calculate_fair_value(
                product, expected_total_sum, std_dev_total_sum
            )

            if fair_value is None:
                continue

            product_type = product.id.split(",")[1]

            if product_type == "F":
                half_spread = future_half_spread

            elif product_type in {"C", "P"}:
                half_spread = max(0.25, fair_value * 0.25)

            else:
                continue
            # Get your current position in this product
            position = my_trades.get_position(product.id)
            numeric_position = 0

            if position is not None:
                numeric_position = position.position
                
            skew_per_unit = half_spread * 0.02
            skew = skew_per_unit * (numeric_position * -1)

            max_skew = half_spread * 0.9
            skew = max(-max_skew, min(max_skew, skew))

            skewed_fv = fair_value + skew

            bid = max(0.1, skewed_fv - half_spread)
            ask = skewed_fv + half_spread

            quotes[product.id] = (bid, ask)

        return quotes
    
    def _get_roll_distribution_params(self, training_rolls, current_rolls, current_sub_round: int) -> Tuple[float, float]:
        if current_sub_round == 10:
            all_rolls = list(current_rolls)
        else:
            all_rolls = list(training_rolls) + list(current_rolls)
        
        if not all_rolls:
            mean = (1 + self.dice_sides) / 2.0
            std_dev = np.sqrt(((self.dice_sides - 1 + 1) ** 2 - 1) / 12)

            return mean, std_dev
        
        return np.mean(all_rolls), np.std(all_rolls)
    
    def _calculate_fair_value(self, product, expected_total_sum: float, std_dev_total_sum: float) -> float:
        """Calculate fair value for a product."""
        try:
            data = product.id.split(",")
            product_type = data[1]
            
            if product_type == "F":  # Future
                return expected_total_sum
                
            elif data[1] in {"C", "P"}:  # Options
                strike_price = float(data[3])
                
                if product_type == "C":
                    return self._price_call(
                        S=expected_total_sum,
                        K=strike_price,
                        sigma=std_dev_total_sum
                    )
                
                else:
                    return self._price_put(
                        S=expected_total_sum,
                        K=strike_price,
                        sigma=std_dev_total_sum
                    )
                
        except (ValueError, IndexError):
            # Malformed product ID
            return None
            
        return None
      
    def _price_call(self, S: float, K: float, sigma: float) -> float:
        if sigma == 0:
            return max(0.0, S - K)
        
        d = (S - K) / sigma
        return (S - K) * norm.cdf(d) + sigma * norm.pdf(d)

    def _price_put(self, S: float, K: float, sigma: float) -> float:
        if sigma == 0:
            return max(0.0, K - S)
            
        d = (S - K) / sigma
        return (K - S) * norm.cdf(-d) + sigma * norm.pdf(-d)
    
    def on_round_end(self, result: Dict[str, Any]) -> None:
        """Handle end of round printing for personal debugging."""
        pnl = result.get("pnl", 0.0)
        dice_rolls = result.get("dice_rolls", [])
        print(f"Round ended. PnL: ${pnl:.2f}, Dice: {dice_rolls[:10]}")
        
    def on_game_end(self, summary: Dict[str, Any]) -> None:
        """Handle end of game debugging."""
        total_pnl = summary.get("total_pnl", 0.0)
        final_score = summary.get("final_score", 0.0)
        print(f"Game ended. Total PnL: ${total_pnl:.2f}, Score: {final_score:.1f}")
    