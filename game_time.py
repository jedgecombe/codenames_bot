import logging

from utils.linker import Linker

logging.basicConfig(
    format='%(asctime)s.%(msecs)03d - %(name)s:%(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

ln = Linker()
ln.update_game_words("root, sound, field, bear, loch ness, spike, part, robot, angel", category='link', action='add')
ln.update_game_words("note, princess, tap, alps, row, crown, genius, code",  category='avoid', action='add')
ln.update_game_words("pound",  category='assassin', action='add')
ln.hypernym_taxonomy(max_depth=3, assassin_limit=0.2, opposition_limit=0.05, neutral_limit=0.025)
# ln.construct_comparisons_df()
ln.recommendation_gen(min_similarity=0.75)
