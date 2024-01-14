import wikipedia
from langdetect import detect
import string

def clean_text(text):
   # Create a translation table that maps every character to itself
   translator = str.maketrans({chr(i): chr(i) for i in range(256)})
   
   # Overwrite the mappings for non-letter and non-space characters
   for char in string.punctuation + string.digits:
       translator[ord(char)] = ""
   
   cleaned_text = text.translate(translator).lower()
   return cleaned_text

def get_random_phrase(language_code, max_words=30):
    try:
        wikipedia.set_lang(language_code)
        page_title = wikipedia.random()
        page = wikipedia.page(page_title)
        raw_text = clean_text(page.content)
        # Limit to max_words
        words = raw_text.split()[:max_words]
        return ' '.join(words)
    except Exception as e:
        print(f"Error fetching random phrase in {language_code}: {e}")
        return None

def detect_language(phrase):
    try:
        language_code = detect(phrase)
        return language_code
    except Exception as e:
        print(f"Error detecting language: {e}")
        return None

def main():
    # output_file = "eng_nl_texts_train.txt"

    # with open(output_file, "w", encoding="utf-8") as file:
    #     for lang_code in ["en", "nl"]:
    #         for _ in range(2000):  # Change the number as needed
    #             random_phrase = get_random_phrase(lang_code)
    #             if random_phrase:
    #                 file.write(f"{lang_code}|{random_phrase}\n")
       
    output_file = "eng_nl_texts_test.txt"

    with open(output_file, "w", encoding="utf-8") as file:
        for lang_code in ["en", "nl"]:
            for _ in range(200):  # Change the number as needed
                random_phrase = get_random_phrase(lang_code)
                if random_phrase:
                    file.write(f"{lang_code}|{random_phrase}\n")             
                    
if __name__ == "__main__":
    main()
