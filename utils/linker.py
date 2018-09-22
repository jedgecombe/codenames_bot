import logging

from tabulate import tabulate

from nltk.corpus import wordnet
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


class Linker:
    def __init__(self):
        self.link_words = []
        self.avoid_words = []
        self.neutral_words = []
        self.assassin_word = []

        self.comparisons_made = []
        self.comparisons_df = pd.DataFrame
        self.recommendation_cnt = 1

    @staticmethod
    def _word_split_check(word_string: str):
        if ',' in word_string:
            words_adj = [x.strip() for x in word_string.split(',')]

        else:
            words_adj = [word_string]
        return words_adj

    def update_game_words(self, word: str, category: str, action: str):
        assert category in ['link', 'avoid', 'neutral', 'assassin'], "category arg should be one of the following:" \
                                                                     "['link', 'avoid', 'neutral', 'assassin']"
        assert action in ['add', 'remove'], "action arg should be either 'add' or 'remove'"

        # TODO - make this more succinct
        if category == 'link':
            if action == 'add':
                self.link_words.extend(self._word_split_check(word))
            elif action == 'remove':
                self.link_words.remove(self._word_split_check(word))
            logger.info(f'{category} words after update: {self.link_words}')
        elif category == 'avoid':
            if action == 'add':
                self.avoid_words.extend(self._word_split_check(word))
            elif action == 'remove':
                self.avoid_words.remove(self._word_split_check(word))
            logger.info(f'{category} words after update: {self.avoid_words}')
        elif category == 'neutral':
            if action == 'add':
                self.neutral_words.extend(self._word_split_check(word))
            elif action == 'remove':
                self.neutral_words.remove(self._word_split_check(word))
            logger.info(f'{category} words after update: {self.neutral_words}')
        elif category == 'assassin':
            if action == 'add':
                self.assassin_word.extend(self._word_split_check(word))
            elif action == 'remove':
                self.assassin_word.remove(self._word_split_check(word))
            logger.info(f'{category} words after update: {self.assassin_word}')

    def avoid_check(self, check_synset, comparison_score: float, assassin_limit: float,
                    opposition_limit: float, neutral_limit: float) -> bool:
        """get a list of words to avoid so that we don't suggest ones that have a high likelihood of leading to
        incorrect guesses"""
        def _avoid_check(synset1, comparison_words: list, limit: float) -> float:
            avoid = False
            for word in comparison_words:
                syn = wordnet.synsets(word)[0]
                # for syn in syns:
                    # logger.debug(f'avoidance check between {check_synset} and {word} (synset: {')
                sim_score = synset1.wup_similarity(syn)
                if sim_score is None:
                    continue
                if sim_score > limit:
                    logger.debug(f'avoid - {synset1} too similar to {syn}, from word: {word} '
                                 f'(similarity score of {round(sim_score, 2)}). Not using.')
                    avoid = True
                    break
            return avoid

        assert 1 >= assassin_limit > opposition_limit > neutral_limit > 0, 'check your limits, should follow: ' \
                                                                           '1 >= assassin_limit > opposition_limit > ' \
                                                                           'neutral_limit > 0'

        is_appropriate = True
        for comp_list, lim in zip([self.assassin_word, self.avoid_words, self.neutral_words], [assassin_limit, opposition_limit, neutral_limit]):
            max_sim = comparison_score - lim
            avoid_check = _avoid_check(check_synset, comp_list, max_sim)
            if avoid_check:
                is_appropriate = False
                break
        return is_appropriate

    def hypernym_taxonomy(self, max_depth: int=2, assassin_limit: float=0.1, opposition_limit: float=0.05,
                          neutral_limit: float=0.025):
        # loop through link words as below
        # find if the best options, compare these against the avoid words etc to see how similar they are, if they are similar, remove that entry
        for w in self.link_words:
            logger.info(f'\n\n attempting to link: {w}')
            # TODO maybe more logical to create a new list and delete rather than in line for lopp
            # find words to compare against
            other_words = [other for other in self.link_words if other != w]
            syns = wordnet.synsets(w)

            sim_check = False

            # for each link word, check against synonyms
            for subject_syn in syns[:max_depth]:
                # check each synonym against other link words (and their synonyms)
                for comp in other_words:
                    logger.info(f'attempting to link with: {comp}')
                    syns2 = wordnet.synsets(comp)
                    for comp_syn in syns2[:max_depth]:
                        comparison_dict = {'subject_word': w, 'comparison_word': comp}
                        lch = subject_syn.common_hypernyms(comp_syn)
                        common_hyp_root_dist = [s.max_depth() for s in lch]
                        sorted_lch = [x for _, x in sorted(zip(common_hyp_root_dist, lch), reverse=True)]

                        if len(sorted_lch) > 0:
                            for option in sorted_lch:
                                wup1 = subject_syn.wup_similarity(option)
                                wup2 = comp_syn.wup_similarity(option)
                                wup_avg = np.mean([wup1, wup2]).item()
                                logger.debug(f'checking appropriateness of: {option} (avg similarity: '
                                             f'{round(wup_avg, 2)})')
                                sim_check = self.avoid_check(option, comparison_score=wup_avg,
                                                             assassin_limit=assassin_limit,
                                                             opposition_limit=opposition_limit,
                                                             neutral_limit=neutral_limit)
                                # if it passes the check
                                if sim_check:
                                    logger.info(f'lowest sufficient hypernym between: {subject_syn} and {comp_syn} is {option}'
                                                f'\n similarity between {subject_syn} and {option} = {round(wup1, 2)}'
                                                f'\n similarity between {comp_syn} and {option} = {round(wup2, 2)}'
                                                f'\n average similarity: {round(wup_avg, 2)}')
                                    comparison_dict['subject_synonym'] = subject_syn
                                    comparison_dict['comparison_synonym'] = comp_syn
                                    comparison_dict['lowest_common_hypernym'] = option
                                    comparison_dict['wup1'] = wup1
                                    comparison_dict['wup2'] = wup2
                                    comparison_dict['wup_avg'] = wup_avg
                                    self.comparisons_made.append(comparison_dict)
                                    break
                        else:
                            comparison_dict['subject_synonym'] = subject_syn
                            comparison_dict['comparison_synonym'] = comp_syn
                            comparison_dict['lowest_common_hypernym'] = ''
                            comparison_dict['wup1'] = 0
                            comparison_dict['wup2'] = 0
                            comparison_dict['wup_avg'] = 0
                            self.comparisons_made.append(comparison_dict)

                        if sim_check:
                            break
                        else:
                            logger.debug(f'no suitable common hypernym found between: {subject_syn} and {comp_syn}')

    def construct_comparisons_df(self):
        self.comparisons_df = pd.DataFrame(self.comparisons_made)

    def recommendation_gen(self, min_similarity: float):
        self.construct_comparisons_df()
        all_recs = self.comparisons_df.copy()
        good_recs = all_recs[all_recs['wup_avg'] >= min_similarity]

        recommendations_made = []

        if len(good_recs) > 0:
            grp = good_recs.groupby(['lowest_common_hypernym']).size()
            recommendation_order = grp.sort_values(ascending=False)
            for ind, row in recommendation_order.iteritems():
                chosen_recommendation = good_recs[good_recs['lowest_common_hypernym'] == ind]
                logger.info(f'\n recommendation {self.recommendation_cnt}: \n'
                            f'use {chosen_recommendation["lowest_common_hypernym"].iloc[0]}'
                            f'to link {chosen_recommendation["comparison_word"].unique()}')
                self.recommendation_cnt += 1
                recommendations_made.append(chosen_recommendation['subject_word'].iloc[0])

        # for single recommendations, use just a synonym
        # NEXT - problem is that it doesn't add ones without a recommendation  to the dataframe
        for word in all_recs['subject_word'].unique():

            if word not in recommendations_made:
                syns = wordnet.synsets(word)
                syn = syns[0]
                for l in syn.lemmas():
                    if l.name()[:2] not in word:
                        logger.info(f'\n recommendation {self.recommendation_cnt}: \n'
                                    f'use {l.name()} to link {word}')
                        self.recommendation_cnt += 1
                        recommendations_made.append(word)
                        break
                    else:
                        logger.debug(f'not linking: {l.name()} with {word}')
