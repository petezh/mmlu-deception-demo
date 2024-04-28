from os.path import join

TEMPLATES_FOLDER = "templates"

def get_template(template_name: str) -> str:
    with open(join(TEMPLATES_FOLDER, template_name + ".txt"), "r") as f:
        return f.read()