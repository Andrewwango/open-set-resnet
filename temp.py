import shutil, os, random


for source in next(os.walk('.'))[1]:
    print(source)
    if source=="Comb": continue
    dest = r'Comb'
    files = os.listdir(source)
    no_of_files = 10

    for file_name in random.sample(files, no_of_files):
        shutil.copy(os.path.join(source, file_name), os.path.join(dest, str(source) + "_" + file_name))