"""Localized fixed strings for bot replies (OTP, final message, etc.). Fallback to English."""
from typing import Any

# Key -> { lang_code -> format string }. Use {0}, {1} for placeholders (email, phone, url).
TEMPLATES: dict[str, dict[str, str]] = {
    "otp_sent_email": {
        "en": "Great! I've sent a 6-digit verification code to {0}. Please enter the code to verify your email.",
        "es": "¡Listo! He enviado un código de verificación de 6 dígitos a {0}. Por favor ingresa el código para verificar tu correo.",
        "hi": "मैंने {0} पर 6 अंकों का सत्यापन कोड भेज दिया है। कृपया अपना ईमेल सत्यापित करने के लिए कोड दर्ज करें।",
        "ur": "میں نے {0} پر 6 ہندسوں کی تصدیقی کوڈ بھیج دی ہے۔ براہ کرم اپنا ای میل تصدیق کرنے کے لیے کوڈ درج کریں۔",
        "pa": "ਮੈਂ {0} 'ਤੇ 6 ਅੰਕਾਂ ਦਾ ਤਸਦੀਕ ਕੋਡ ਭੇਜ ਦਿੱਤਾ ਹੈ। ਕਿਰਪਾ ਕਰਕੇ ਆਪਣੀ ਈਮੇਲ ਤਸਦੀਕ ਕਰਨ ਲਈ ਕੋਡ ਦਾਖਲ ਕਰੋ।",
        "fr": "J'ai envoyé un code de vérification à 6 chiffres à {0}. Entrez le code pour vérifier votre e-mail.",
        "de": "Ich habe einen 6-stelligen Bestätigungscode an {0} gesendet. Bitte geben Sie den Code zur Bestätigung Ihrer E-Mail ein.",
    },
    "otp_sent_phone": {
        "en": "Great! I've sent a 6-digit verification code to {0}. Please enter the code to verify your phone number.",
        "es": "¡Listo! He enviado un código de verificación de 6 dígitos al {0}. Por favor ingresa el código para verificar tu número.",
        "hi": "मैंने {0} पर 6 अंकों का सत्यापन कोड भेज दिया है। कृपया अपना फोन नंबर सत्यापित करने के लिए कोड दर्ज करें।",
        "ur": "میں نے {0} پر 6 ہندسوں کی تصدیقی کوڈ بھیج دی ہے۔ براہ کرم اپنا فون نمبر تصدیق کرنے کے لیے کوڈ درج کریں۔",
        "pa": "ਮੈਂ {0} 'ਤੇ 6 ਅੰਕਾਂ ਦਾ ਤਸਦੀਕ ਕੋਡ ਭੇਜ ਦਿੱਤਾ ਹੈ। ਕਿਰਪਾ ਕਰਕੇ ਆਪਣਾ ਫੋਨ ਨੰਬਰ ਤਸਦੀਕ ਕਰਨ ਲਈ ਕੋਡ ਦਾਖਲ ਕਰੋ।",
        "fr": "J'ai envoyé un code de vérification à 6 chiffres au {0}. Entrez le code pour vérifier votre numéro.",
        "de": "Ich habe einen 6-stelligen Bestätigungscode an {0} gesendet. Bitte geben Sie den Code zur Bestätigung Ihrer Nummer ein.",
    },
    "otp_resend": {
        "en": "I've sent a new verification code to {0}. Please enter the code.",
        "es": "He enviado un nuevo código de verificación a {0}. Por favor ingresa el código.",
        "hi": "मैंने {0} पर एक नया सत्यापन कोड भेज दिया है। कृपया कोड दर्ज करें।",
        "ur": "میں نے {0} پر نیا تصدیقی کوڈ بھیج دیا ہے۔ براہ کرم کوڈ درج کریں۔",
        "pa": "ਮੈਂ {0} 'ਤੇ ਨਵਾਂ ਤਸਦੀਕ ਕੋਡ ਭੇਜ ਦਿੱਤਾ ਹੈ। ਕਿਰਪਾ ਕਰਕੇ ਕੋਡ ਦਾਖਲ ਕਰੋ।",
        "fr": "J'ai envoyé un nouveau code de vérification à {0}. Entrez le code.",
        "de": "Ich habe einen neuen Bestätigungscode an {0} gesendet. Bitte geben Sie den Code ein.",
    },
    "perfect_otp_sent_email": {
        "en": "Perfect! I've sent a verification code to {0}. Please enter the code to verify your email.",
        "es": "¡Perfecto! He enviado un código de verificación a {0}. Por favor ingresa el código para verificar tu correo.",
        "hi": "मैंने {0} पर सत्यापन कोड भेज दिया है। कृपया अपना ईमेल सत्यापित करने के लिए कोड दर्ज करें।",
        "ur": "میں نے {0} پر تصدیقی کوڈ بھیج دیا ہے۔ براہ کرم اپنا ای میل تصدیق کرنے کے لیے کوڈ درج کریں۔",
        "pa": "ਮੈਂ {0} 'ਤੇ ਤਸਦੀਕ ਕੋਡ ਭੇਜ ਦਿੱਤਾ ਹੈ। ਕਿਰਪਾ ਕਰਕੇ ਆਪਣੀ ਈਮੇਲ ਤਸਦੀਕ ਕਰਨ ਲਈ ਕੋਡ ਦਾਖਲ ਕਰੋ।",
        "fr": "J'ai envoyé un code de vérification à {0}. Entrez le code pour vérifier votre e-mail.",
        "de": "Ich habe einen Bestätigungscode an {0} gesendet. Bitte geben Sie den Code zur Bestätigung Ihrer E-Mail ein.",
    },
    "perfect_otp_sent_phone": {
        "en": "Perfect! I've sent a verification code to {0}. Please enter the code to verify your phone number.",
        "es": "¡Perfecto! He enviado un código de verificación al {0}. Por favor ingresa el código para verificar tu número.",
        "hi": "मैंने {0} पर सत्यापन कोड भेज दिया है। कृपया अपना फोन नंबर सत्यापित करने के लिए कोड दर्ज करें।",
        "ur": "میں نے {0} پر تصدیقی کوڈ بھیج دیا ہے۔ براہ کرم اپنا فون نمبر تصدیق کرنے کے لیے کوڈ درج کریں۔",
        "pa": "ਮੈਂ {0} 'ਤੇ ਤਸਦੀਕ ਕੋਡ ਭੇਜ ਦਿੱਤਾ ਹੈ। ਕਿਰਪਾ ਕਰਕੇ ਆਪਣਾ ਫੋਨ ਨੰਬਰ ਤਸਦੀਕ ਕਰਨ ਲਈ ਕੋਡ ਦਾਖਲ ਕਰੋ।",
        "fr": "J'ai envoyé un code de vérification au {0}. Entrez le code pour vérifier votre numéro.",
        "de": "Ich habe einen Bestätigungscode an {0} gesendet. Bitte geben Sie den Code zur Bestätigung Ihrer Nummer ein.",
    },
    "no_problem_email": {
        "en": "No problem! Please provide your correct email address.",
        "es": "¡No hay problema! Por favor proporciona tu correo electrónico correcto.",
        "hi": "कोई बात नहीं! कृपया अपना सही ईमेल पता दें।",
        "ur": "کوئی بات نہیں! براہ کرم اپنا درست ای میل پتہ فراہم کریں۔",
        "pa": "ਕੋਈ ਗੱਲ ਨਹੀਂ! ਕਿਰਪਾ ਕਰਕੇ ਆਪਣਾ ਸਹੀ ਈਮੇਲ ਪਤਾ ਦਿਓ।",
        "fr": "Pas de problème ! Veuillez fournir votre adresse e-mail correcte.",
        "de": "Kein Problem! Bitte geben Sie Ihre richtige E-Mail-Adresse an.",
    },
    "no_problem_phone": {
        "en": "No problem! Please provide your correct phone number.",
        "es": "¡No hay problema! Por favor proporciona tu número de teléfono correcto.",
        "hi": "कोई बात नहीं! कृपया अपना सही फोन नंबर दें।",
        "ur": "کوئی بات نہیں! براہ کرم اپنا درست فون نمبر فراہم کریں۔",
        "pa": "ਕੋਈ ਗੱਲ ਨਹੀਂ! ਕਿਰਪਾ ਕਰਕੇ ਆਪਣਾ ਸਹੀ ਫੋਨ ਨੰਬਰ ਦਿਓ।",
        "fr": "Pas de problème ! Veuillez fournir votre numéro de téléphone correct.",
        "de": "Kein Problem! Bitte geben Sie Ihre richtige Telefonnummer an.",
    },
    "final_success": {
        "en": "Thanks! I have your details and someone will get back to you soon. Bye!",
        "es": "¡Gracias! Tengo tus datos y alguien se pondrá en contacto pronto. ¡Hasta luego!",
        "hi": "धन्यवाद! मेरे पास आपका विवरण है और जल्द ही कोई संपर्क करेगा। अलविदा!",
        "ur": "شکریہ! میرے پاس آپ کی تفصیلات ہیں اور جلد کوئی آپ سے رابطہ کرے گا۔ خدا حافظ!",
        "pa": "ਧੰਨਵਾਦ! ਮੇਰੇ ਕੋਲ ਤੁਹਾਡੇ ਵੇਰਵੇ ਹਨ ਅਤੇ ਜਲਦੀ ਕੋਈ ਸੰਪਰਕ ਕਰੇਗਾ। ਅਲਵਿਦਾ!",
        "fr": "Merci ! J'ai vos coordonnées et quelqu'un vous recontactera bientôt. Au revoir !",
        "de": "Danke! Ich habe Ihre Daten und jemand wird sich bald bei Ihnen melden. Tschüss!",
    },
    "final_fallback": {
        "en": "Thanks! I captured your details. There was a small issue creating the lead right now, but the team will still follow up shortly. Bye!",
        "es": "¡Gracias! Guardé tus datos. Hubo un pequeño problema al crear el registro ahora, pero el equipo te contactará pronto. ¡Hasta luego!",
        "hi": "धन्यवाद! मैंने आपका विवरण सहेज लिया। अभी लीड बनाने में थोड़ी समस्या हुई, लेकिन टीम जल्द ही संपर्क करेगी। अलविदा!",
        "ur": "شکریہ! میں نے آپ کی تفصیلات محفوظ کر لیں۔ ابھی لیڈ بنانے میں تھوڑی مسئلہ آئی، لیکن ٹیم جلد رابطہ کرے گی۔ خدا حافظ!",
        "pa": "ਧੰਨਵਾਦ! ਮੈਂ ਤੁਹਾਡੇ ਵੇਰਵੇ ਸੇਵ ਕਰ ਲਏ। ਹੁਣ ਲੀਡ ਬਣਾਉਣ ਵਿੱਚ ਥੋੜ੍ਹੀ ਸਮੱਸਿਆ ਆਈ, ਪਰ ਟੀਮ ਜਲਦੀ ਸੰਪਰਕ ਕਰੇਗੀ। ਅਲਵਿਦਾ!",
        "fr": "Merci ! J'ai enregistré vos coordonnées. Un petit problème est survenu lors de la création du lead, mais l'équipe vous recontactera bientôt. Au revoir !",
        "de": "Danke! Ich habe Ihre Daten erfasst. Beim Erstellen des Leads ist ein kleines Problem aufgetreten, aber das Team wird sich trotzdem bald bei Ihnen melden. Tschüss!",
    },
    "review_prompt": {
        "en": "We'd love to hear from you! If you have a moment, please leave us a review on Google: {0}",
        "es": "¡Nos encantaría saber tu opinión! Si tienes un momento, deja una reseña en Google: {0}",
        "hi": "हम आपसे सुनना पसंद करेंगे! अगर आपके पास समय है, तो कृपया Google पर समीक्षा दें: {0}",
        "ur": "ہم آپ کی رائے سننا پسند کریں گے! اگر آپ کے پاس وقت ہو تو براہ کرم گوگل پر ہمیں ریویو دیں: {0}",
        "pa": "ਅਸੀਂ ਤੁਹਾਡੀ ਰਾਏ ਸੁਣਨਾ ਪਸੰਦ ਕਰਾਂਗੇ! ਜੇਕਰ ਤੁਹਾਡੇ ਕੋਲ ਸਮਾਂ ਹੈ ਤਾਂ ਕਿਰਪਾ ਕਰਕੇ ਗੂਗਲ 'ਤੇ ਸਮੀਖਿਆ ਦਿਓ: {0}",
        "fr": "Nous aimerions avoir votre avis ! Si vous avez un moment, laissez-nous un avis sur Google : {0}",
        "de": "Wir freuen uns über Ihr Feedback! Wenn Sie einen Moment Zeit haben, hinterlassen Sie uns eine Bewertung auf Google: {0}",
    },
    "otp_send_fail_email": {
        "en": "Sorry, I couldn't send the verification email. Please check your email address and try again.",
        "es": "Lo siento, no pude enviar el correo de verificación. Por favor verifica tu correo e intenta de nuevo.",
        "hi": "क्षमा करें, मैं सत्यापन ईमेल नहीं भेज सका। कृपया अपना ईमेल पता जांचें और पुनः प्रयास करें।",
        "ur": "معذرت، میں تصدیقی ای میل نہیں بھیج سکا۔ براہ کرم اپنا ای میل پتہ چیک کریں اور دوبارہ کوشش کریں۔",
        "pa": "ਮਾਫ਼ ਕਰਨਾ, ਮੈਂ ਤਸਦੀਕ ਈਮੇਲ ਨਹੀਂ ਭੇਜ ਸਕਿਆ। ਕਿਰਪਾ ਕਰਕੇ ਆਪਣਾ ਈਮੇਲ ਪਤਾ ਜਾਂਚੋ ਅਤੇ ਦੁਬਾਰਾ ਕੋਸ਼ਿਸ਼ ਕਰੋ।",
        "fr": "Désolé, je n'ai pas pu envoyer l'e-mail de vérification. Vérifiez votre adresse e-mail et réessayez.",
        "de": "Entschuldigung, ich konnte die Bestätigungs-E-Mail nicht senden. Bitte überprüfen Sie Ihre E-Mail-Adresse und versuchen Sie es erneut.",
    },
    "otp_send_fail_phone": {
        "en": "Sorry, I couldn't send the verification SMS. Please check your phone number and try again.",
        "es": "Lo siento, no pude enviar el SMS de verificación. Por favor verifica tu número e intenta de nuevo.",
        "hi": "क्षमा करें, मैं सत्यापन एसएमएस नहीं भेज सका। कृपया अपना फोन नंबर जांचें और पुनः प्रयास करें।",
        "ur": "معذرت، میں تصدیقی ایس ایم ایس نہیں بھیج سکا۔ براہ کرم اپنا فون نمبر چیک کریں اور دوبارہ کوشش کریں۔",
        "pa": "ਮਾਫ਼ ਕਰਨਾ, ਮੈਂ ਤਸਦੀਕ ਐਸਐਮਐਸ ਨਹੀਂ ਭੇਜ ਸਕਿਆ। ਕਿਰਪਾ ਕਰਕੇ ਆਪਣਾ ਫੋਨ ਨੰਬਰ ਜਾਂਚੋ ਅਤੇ ਦੁਬਾਰਾ ਕੋਸ਼ਿਸ਼ ਕਰੋ।",
        "fr": "Désolé, je n'ai pas pu envoyer le SMS de vérification. Vérifiez votre numéro et réessayez.",
        "de": "Entschuldigung, ich konnte die Bestätigungs-SMS nicht senden. Bitte überprüfen Sie Ihre Nummer und versuchen Sie es erneut.",
    },
    "otp_resend_fail_email": {
        "en": "Sorry, I couldn't resend the verification email. Please try again.",
        "es": "Lo siento, no pude reenviar el correo de verificación. Por favor intenta de nuevo.",
        "hi": "क्षमा करें, मैं सत्यापन ईमेल पुनः नहीं भेज सका। कृपया पुनः प्रयास करें।",
        "ur": "معذرت، میں تصدیقی ای میل دوبارہ نہیں بھیج سکا۔ براہ کرم دوبارہ کوشش کریں۔",
        "pa": "ਮਾਫ਼ ਕਰਨਾ, ਮੈਂ ਤਸਦੀਕ ਈਮੇਲ ਦੁਬਾਰਾ ਨਹੀਂ ਭੇਜ ਸਕਿਆ। ਕਿਰਪਾ ਕਰਕੇ ਦੁਬਾਰਾ ਕੋਸ਼ਿਸ਼ ਕਰੋ।",
        "fr": "Désolé, je n'ai pas pu renvoyer l'e-mail de vérification. Veuillez réessayer.",
        "de": "Entschuldigung, ich konnte die Bestätigungs-E-Mail nicht erneut senden. Bitte versuchen Sie es erneut.",
    },
    "otp_resend_fail_phone": {
        "en": "Sorry, I couldn't resend the verification SMS. Please try again.",
        "es": "Lo siento, no pude reenviar el SMS de verificación. Por favor intenta de nuevo.",
        "hi": "क्षमा करें, मैं सत्यापन एसएमएस पुनः नहीं भेज सका। कृपया पुनः प्रयास करें।",
        "ur": "معذرت، میں تصدیقی ایس ایم ایس دوبارہ نہیں بھیج سکا۔ براہ کرم دوبارہ کوشش کریں۔",
        "pa": "ਮਾਫ਼ ਕਰਨਾ, ਮੈਂ ਤਸਦੀਕ ਐਸਐਮਐਸ ਦੁਬਾਰਾ ਨਹੀਂ ਭੇਜ ਸਕਿਆ। ਕਿਰਪਾ ਕਰਕੇ ਦੁਬਾਰਾ ਕੋਸ਼ਿਸ਼ ਕਰੋ।",
        "fr": "Désolé, je n'ai pas pu renvoyer le SMS de vérification. Veuillez réessayer.",
        "de": "Entschuldigung, ich konnte die Bestätigungs-SMS nicht erneut senden. Bitte versuchen Sie es erneut.",
    },
    "no_email": {
        "en": "I don't have your email address. Please provide your email address.",
        "es": "No tengo tu correo electrónico. Por favor proporciónalo.",
        "hi": "मेरे पास आपका ईमेल पता नहीं है। कृपया अपना ईमेल पता दें।",
        "ur": "میرے پاس آپ کا ای میل پتہ نہیں ہے۔ براہ کرم اپنا ای میل پتہ فراہم کریں۔",
        "pa": "ਮੇਰੇ ਕੋਲ ਤੁਹਾਡਾ ਈਮੇਲ ਪਤਾ ਨਹੀਂ ਹੈ। ਕਿਰਪਾ ਕਰਕੇ ਆਪਣਾ ਈਮੇਲ ਪਤਾ ਦਿਓ।",
        "fr": "Je n'ai pas votre adresse e-mail. Veuillez la fournir.",
        "de": "Ich habe Ihre E-Mail-Adresse nicht. Bitte geben Sie sie an.",
    },
    "no_phone": {
        "en": "I don't have your phone number. Please provide your phone number.",
        "es": "No tengo tu número de teléfono. Por favor proporciónalo.",
        "hi": "मेरे पास आपका फोन नंबर नहीं है। कृपया अपना फोन नंबर दें।",
        "ur": "میرے پاس آپ کا فون نمبر نہیں ہے۔ براہ کرم اپنا فون نمبر فراہم کریں۔",
        "pa": "ਮੇਰੇ ਕੋਲ ਤੁਹਾਡਾ ਫੋਨ ਨੰਬਰ ਨਹੀਂ ਹੈ। ਕਿਰਪਾ ਕਰਕੇ ਆਪਣਾ ਫੋਨ ਨੰਬਰ ਦਿਓ।",
        "fr": "Je n'ai pas votre numéro de téléphone. Veuillez le fournir.",
        "de": "Ich habe Ihre Telefonnummer nicht. Bitte geben Sie sie an.",
    },
    "otp_wrong_code": {
        "en": "That code doesn't look right. Please check and try entering the 6-digit code again.",
        "es": "Ese código no parece correcto. Por favor verifica e ingresa el código de 6 dígitos de nuevo.",
        "hi": "वह कोड सही नहीं लगता। कृपया 6 अंकों का कोड फिर से दर्ज करें।",
        "ur": "وہ کوڈ درست نہیں لگتا۔ براہ کرم 6 ہندسوں کا کوڈ دوبارہ درج کریں۔",
        "pa": "ਉਹ ਕੋਡ ਸਹੀ ਨਹੀਂ ਲੱਗਦਾ। ਕਿਰਪਾ ਕਰਕੇ 6 ਅੰਕਾਂ ਦਾ ਕੋਡ ਦੁਬਾਰਾ ਦਾਖਲ ਕਰੋ।",
        "fr": "Ce code ne semble pas correct. Vérifiez et réessayez avec le code à 6 chiffres.",
        "de": "Dieser Code scheint nicht richtig zu sein. Bitte überprüfen Sie und geben Sie den 6-stelligen Code erneut ein.",
    },
    "otp_please_enter": {
        "en": "Please enter the 6-digit verification code.",
        "es": "Por favor ingresa el código de verificación de 6 dígitos.",
        "hi": "कृपया 6 अंकों का सत्यापन कोड दर्ज करें।",
        "ur": "براہ کرم 6 ہندسوں کا تصدیقی کوڈ درج کریں۔",
        "pa": "ਕਿਰਪਾ ਕਰਕੇ 6 ਅੰਕਾਂ ਦਾ ਤਸਦੀਕ ਕੋਡ ਦਾਖਲ ਕਰੋ।",
        "fr": "Veuillez entrer le code de vérification à 6 chiffres.",
        "de": "Bitte geben Sie den 6-stelligen Bestätigungscode ein.",
    },
    "found_email_cant_send": {
        "en": "I found your email, but couldn't send the verification code. Please try again.",
        "es": "Encontré tu correo, pero no pude enviar el código de verificación. Por favor intenta de nuevo.",
        "hi": "मुझे आपका ईमेल मिल गया, लेकिन मैं सत्यापन कोड नहीं भेज सका। कृपया पुनः प्रयास करें।",
        "ur": "مجھے آپ کا ای میل ملا لیکن میں تصدیقی کوڈ نہیں بھیج سکا۔ براہ کرم دوبارہ کوشش کریں۔",
        "pa": "ਮੈਨੂੰ ਤੁਹਾਡੀ ਈਮੇਲ ਮਿਲੀ ਪਰ ਮੈਂ ਤਸਦੀਕ ਕੋਡ ਨਹੀਂ ਭੇਜ ਸਕਿਆ। ਕਿਰਪਾ ਕਰਕੇ ਦੁਬਾਰਾ ਕੋਸ਼ਿਸ਼ ਕਰੋ।",
        "fr": "J'ai trouvé votre e-mail mais je n'ai pas pu envoyer le code. Veuillez réessayer.",
        "de": "Ich habe Ihre E-Mail gefunden, konnte den Bestätigungscode aber nicht senden. Bitte versuchen Sie es erneut.",
    },
    "found_phone_cant_send": {
        "en": "I found your phone number, but couldn't send the verification code. Please try again.",
        "es": "Encontré tu número, pero no pude enviar el código de verificación. Por favor intenta de nuevo.",
        "hi": "मुझे आपका फोन नंबर मिल गया, लेकिन मैं सत्यापन कोड नहीं भेज सका। कृपया पुनः प्रयास करें।",
        "ur": "مجھے آپ کا فون نمبر ملا لیکن میں تصدیقی کوڈ نہیں بھیج سکا۔ براہ کرم دوبارہ کوشش کریں۔",
        "pa": "ਮੈਨੂੰ ਤੁਹਾਡਾ ਫੋਨ ਨੰਬਰ ਮਿਲਿਆ ਪਰ ਮੈਂ ਤਸਦੀਕ ਕੋਡ ਨਹੀਂ ਭੇਜ ਸਕਿਆ। ਕਿਰਪਾ ਕਰਕੇ ਦੁਬਾਰਾ ਕੋਸ਼ਿਸ਼ ਕਰੋ।",
        "fr": "J'ai trouvé votre numéro mais je n'ai pas pu envoyer le code. Veuillez réessayer.",
        "de": "Ich habe Ihre Nummer gefunden, konnte den Bestätigungscode aber nicht senden. Bitte versuchen Sie es erneut.",
    },
}


def get_string(key: str, lang_code: str, *args: Any) -> str:
    """Return localized string for key; fallback to English. Format with args if provided."""
    lang = (lang_code or "en").lower().strip()
    table = TEMPLATES.get(key, {})
    template = table.get(lang) or table.get("en", "")
    if not template:
        return "" if not args else str(args[0])
    if args:
        try:
            return template.format(*args)
        except (IndexError, KeyError):
            return template
    return template
