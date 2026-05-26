from tools.commodity_predictor import CommodityPredictor
import warnings
warnings.filterwarnings('ignore')

p = CommodityPredictor('gold')
if p.fetch_data():
    result = p.full_pipeline()
    print('summary ok=', result is not None)
    if result is not None:
        print('commodity', result['commodity'])
        print('1m', result['predictions']['1_month']['price'])
else:
    print('fetch failed')
