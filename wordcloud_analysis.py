from wordcloud import WordCloud
import pandas as pd
import matplotlib.pyplot as plt  # type: ignore

fake = pd.read_csv("data/Fake.csv")

text = " ".join(fake["text"])

wc = WordCloud(
    width=1000,
    height=500
).generate(text)

plt.imshow(wc)
plt.axis("off")
plt.show()