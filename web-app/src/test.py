from pil_utils import *

diag = get_diagram(model_name="spaniels")
diags = split_diagram(diag)
for i,d in enumerate(diags):
    d.save(f"test{i}.jpg")