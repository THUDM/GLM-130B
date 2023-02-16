LANGUAGE_TAG = {
    "c++"          : "// C++",
    "c"            : "// C",
    "c#"           : "// C#",
    "go"           : "// Go",
    "java"         : "// Java",
    "javascript"   : "// JavaScript",
    "kotlin"       : "// Kotlin",
    "php"          : "// PHP",
    "python"       : "# Python",
    "rust"         : "// Rust",
    "ruby"         : "# Ruby",
    "typescript"   : "// TypeScript",
}

def code_generation_end(code, language):
    if language.lower() == 'python':
        end_words = ["\ndef", "\nclass", "\nif", "\n#", "\nprint", "\nassert"]
        for w in end_words:
            if w in code:
                return True
    
    return False

def cleanup_code(code, language):
    if language.lower() == "python":
        end_words = ["\ndef", "\nclass", "\nif", "\n#", "\nprint", "\nassert"]
        for w in end_words:
            if w in code:
                code = code[:code.rfind(w)].rstrip()
    return code
