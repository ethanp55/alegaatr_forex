from genetics.genome import GeneticFeature, Genome
from strategies.squeeze_pro import SqueezePro
from typing import Dict


class SqueezeProGenome(Genome):
    def __init__(self, currency_pair: str, time_frame: str, year: int) -> None:
        super().__init__(currency_pair, time_frame, year, SqueezePro())

    def _initialize_features(self) -> Dict[str, GeneticFeature]:
        n_reds_feature = GeneticFeature([2, 3, 4, 5, 7, 9])
        ma_key_feature = GeneticFeature([None, 'ema200', 'ema100', 'ema50', 'smma200', 'smma100', 'smma50'])
        invert_feature = GeneticFeature([True, False])
        use_tsl_feature = GeneticFeature([True, False])
        pips_to_risk_feature = GeneticFeature([None, 20, 30, 50, 100])
        pips_to_risk_atr_multiplier_feature = GeneticFeature([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        risk_reward_ratio_feature = GeneticFeature([None, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
        lookback_feature = GeneticFeature([25, 50, 100, 150, 200, 250, 300])
        close_trade_incrementally_feature = GeneticFeature([True, False])

        feature_dictionary = {'n_reds': n_reds_feature, 'ma_key': ma_key_feature,
                              'invert': invert_feature, 'use_tsl': use_tsl_feature,
                              'pips_to_risk': pips_to_risk_feature,
                              'pips_to_risk_atr_multiplier': pips_to_risk_atr_multiplier_feature,
                              'risk_reward_ratio': risk_reward_ratio_feature,
                              'lookback': lookback_feature,
                              'close_trade_incrementally': close_trade_incrementally_feature}

        return feature_dictionary
