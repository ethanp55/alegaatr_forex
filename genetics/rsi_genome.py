from genetics.genome import GeneticFeature, Genome
from strategies.rsi import RSI
from typing import Dict


class RSIGenome(Genome):
    def __init__(self, currency_pair: str, time_frame: str) -> None:
        super().__init__(currency_pair, time_frame, RSI())

    def _initialize_features(self) -> Dict[str, GeneticFeature]:
        ma_key_feature = GeneticFeature([None, 'ema200', 'ema100', 'ema50', 'smma200', 'smma100', 'smma50'])
        invert_feature = GeneticFeature([True, False])
        use_tsl_feature = GeneticFeature([True, False])
        pips_to_risk_feature = GeneticFeature([None, 20, 30, 50, 100])
        pips_to_risk_atr_multiplier_feature = GeneticFeature([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        risk_reward_ratio_feature = GeneticFeature([None, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
        close_trade_incrementally_feature = GeneticFeature([True, False])

        feature_dictionary = {'ma_key': ma_key_feature, 'invert': invert_feature, 'use_tsl': use_tsl_feature,
                              'pips_to_risk': pips_to_risk_feature,
                              'pips_to_risk_atr_multiplier': pips_to_risk_atr_multiplier_feature,
                              'risk_reward_ratio': risk_reward_ratio_feature,
                              'close_trade_incrementally': close_trade_incrementally_feature}

        return feature_dictionary
