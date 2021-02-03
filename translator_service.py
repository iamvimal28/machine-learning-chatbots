from google_trans_new import google_translator  
  
translator = google_translator()

def detect_language(text):
    detected_text = translator.detect(text)
    return detected_text[0]

def translate(text,source,destination):
    translate_text = translator.translate(text, lang_src=source, lang_tgt=destination)  
    return translate_text