import snscrape.modules.twitter as sntwitter

# Corrección 1: Separar la importación de la variable
query = "ley fula de tal"
tweets = [] # Corrección 2: Nombre de variable consistente

for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
    if i > 10:
        break
    tweets.append((tweet.user.username, tweet.content))

for user, content in tweets:
    print(f"Usuario: {user}")
    print(f"Texto: {content}\n")