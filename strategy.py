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
        expected_value_per_roll = self._calculate_expected_roll_value(
            training_rolls, current_rolls, round_info.get("current_sub_round", 0)
        )
        
        quotes = {}

        current_sub_round = round_info.get("current_sub_round", 0)
        self.spread_width = self._update_spread(training_rolls, current_rolls)
        
        # Quote on all available products
        for product in marketplace.get_products():

            fair_value = self._calculate_fair_value(
                product, expected_value_per_roll
            )
            
            # Get your current position in this product
            position = my_trades.get_position(product.id)

            numeric_position = 0
            if position is not None:
                # position.trades: list of trades, each has attributes buyer_id, seller_id, price, 
                # round_traded, quantity (quantity is 1 if buying, -1 if selling)
                trades = position.trades

                # position.position: net position, positive if long, negative if short
                numeric_position = position.position

            if fair_value is not None:
                # Create bid/ask spread around fair value

                # Skew quotes to manage positions (sell if long, buy if short)
                fair_value += (self.spread_width / 10) * (numeric_position * -1)

                half_spread = self.spread_width / 2.0
                bid = max(0.1, fair_value - half_spread)
                ask = fair_value + half_spread


                # Add to quotes dictionary
                quotes[product.id] = (bid, ask)

        return quotes
    
    def _update_spread(self, training_rolls, current_rolls) -> float:
        z_score = 1.645
        return z_score * self._calculate_standard_error_of_mean(training_rolls, current_rolls)
    
    def _calculate_expected_roll_value(self, training_rolls, current_rolls, current_sub_round: int) -> float:
        """Calculate expected value of a single dice roll."""
        if current_sub_round == 10:
            all_rolls = list(current_rolls)

        else:
            all_rolls = list(training_rolls) + list(current_rolls)
            
        if not all_rolls:
            # No data available, use theoretical expected value
            return (1 + self.dice_sides) / 2.0
        
        # Calculate empirical expected value
        return np.mean(all_rolls)
    
    def _calculate_fair_value(self, product, expected_value_per_roll: float) -> float:
        """Calculate fair value for a product."""
        try:
            data = product.id.split(",")
            
            if data[1] == "F":  # Future
                settlement_round = int(data[2])
                # Future settles to sum of first settlement_round dice rolls
                fair_value = expected_value_per_roll * settlement_round * 2_000
                return fair_value
                
            elif data[1] in ["C", "P"]:  # Options
                strike_price = float(data[2])
                expiry_round = int(data[3])
                
                # Expected sum at expiry
                expected_sum_at_expiry = expected_value_per_roll * expiry_round * 2_000
                
                if data[1] == "C":  # Call option
                    # Call value = max(0, expected_sum - strike)
                    fair_value = max(0, expected_sum_at_expiry - strike_price)
                else:  # Put option  
                    # Put value = max(0, strike - expected_sum)
                    fair_value = max(0, strike_price - expected_sum_at_expiry)
                
                return fair_value
                
        except (ValueError, IndexError):
            # Malformed product ID
            return None
            
        return None
    
    def _calculate_standard_error_of_mean(self, training_rolls, current_rolls):
        
        all_rolls = list(training_rolls) + list(current_rolls)
        
        if not all_rolls:
            # No data available, use approximate standard error of mean for discrete uniform distribution
            return np.sqrt(len(all_rolls)) / 3.46
        
        # Calculate empirical standard error of mean
        return np.std(all_rolls) / np.sqrt(len(all_rolls))
        
    
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
    