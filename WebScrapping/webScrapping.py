import snscrape.modules.twitter as sntwitter

# Buscar tweets que contengan la query
query = "ley fula de tal"
tweets = []

for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
    if i > 10:
        break
    tweets.append((tweet.user.username, tweet.content))

for user, content in tweets:
    print(f"Usuario: {user}")
    print(f"Texto: {content}\n")