models = { 
    'GEMINI_3_PRO_PREVIEW' : 'gemini-3-pro-preview',
'GEMINI_2_5_PRO' : 'gemini-2.5-pro',
'GEMINI_2_5_FLASH' : 'gemini-2.5-flash',
'CLAUDE_3_SONNET' :  'claude-3.7-sonnet',
'CLAUDE_4_SONNET' : 'claude_4.5_sonnet',
}


def available_models():
    return models.keys()

def get_model_api_name(model_key):
    return models.get(model_key, None)