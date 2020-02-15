import os, shutil

self_path = os.path.dirname(os.path.abspath(__file__))
target_folder = os.path.join(self_path, "cats")

for path, subdirs, files in os.walk(target_folder):
    for name in files:
        tf_format = name.split(".")
        if len(tf_format) == 2:
            file_format = tf_format[1]
            if file_format == "jpg" or file_format == "png":
                try:
                    shutil.move(os.path.join(path, name), target_folder)
                except shutil.Error as e:
                    print("Error moving %s: %s, renaming..." % (name, e))
        else:
            os.remove(os.path.join(path, name))
    if path!=target_folder:
        shutil.rmtree(path)