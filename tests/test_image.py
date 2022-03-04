
import pandas as pd
from convectors.image import Vectorize
from convectors.image.data import Fetch

model = Fetch(db_type="sqlitedict")
images = model([
    "https://www.numerama.com/wp-content/uploads/2019/11/tumblr-logo.jpg",
    "https://64.media.tumblr.com/e2a577f7e15c879323d5b8ffd483269c/f1fa637034d6c832-a8/s400x600/04481726a0add8f3597591ef35e55c7d0d5174e8.png",
    "ejgoiuaehgoaengpe"])
print(images)

model = Vectorize()

df = pd.DataFrame()
df["features"] = list(model(images))
