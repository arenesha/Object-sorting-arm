# Upload trained YOLO models (.pt or .engine) into this folder.
#
# When you drop a model file here:
# 1. Run 'python jetson/select_model.py' to activate it.
# 2. An auto-generated mapping file '<model_name>_mapping.json' will be created.
# 3. Edit the mapping file to assign bin numbers (1, 2, 3...) to your classes.
#    Classes mapped to 0 are skipped/ignored.
