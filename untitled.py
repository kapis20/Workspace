import random
import string

def generate_random_characters(length=10):
    # Define the character pool
    characters = string.ascii_letters + string.digits + string.punctuation
    # Generate random characters
    random_characters = ''.join(random.choice(characters) for _ in range(length))
    return random_characters

# Generate and print 10 random characters
random_chars = generate_random_characters()
print("Random Characters:", random_chars)