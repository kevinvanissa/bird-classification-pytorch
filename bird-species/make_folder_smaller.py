import os
import shutil

def count_items_in_folder(folder):
    return sum(len(files) for _, _, files in os.walk(folder))

def select_least_items_folder(input_folder, n=10):
    subfolders = [f.path for f in os.scandir(input_folder) if f.is_dir() ]
    # Create a list of tuples (folder_path, item_count)
    folder_item_counts = [(subfolder, count_items_in_folder(subfolder)) for subfolder in subfolders]

    # Sort the list by item count and select the least n folders
    least_items_folders = sorted(folder_item_counts, key=lambda x: x[1])[:n]
    
    return [folder for folder, _ in least_items_folders]

def copy_selected_folders(selected_folders, output_folder_train, input_folder_test_valid, output_folder_test_valid):
    """Copy selected folders to the output folder"""
    os.makedirs(output_folder_train, exist_ok=True)
    os.makedirs(output_folder_test_valid, exist_ok=True)

    for folder in selected_folders:
        folder_name = os.path.basename(folder)
        destination = os.path.join(output_folder_train, folder_name)
        shutil.copytree(folder, destination)

        input_test_valid_dir = os.path.join(input_folder_test_valid, folder_name)
        destination_test_valid = os.path.join(output_folder_test_valid, folder_name)
        shutil.copytree(input_test_valid_dir, destination_test_valid)


def main(input_folder_train, output_folder_train, input_folder_test_valid, output_folder_test_valid):
    selected_folders = select_least_items_folder(input_folder_train, 50)
    #copy_selected_folders(selected_folders, output_folder_train)
    copy_selected_folders(selected_folders, output_folder_train, input_folder_test_valid, output_folder_test_valid)


if __name__ == "__main__":
    img_dir_train = 'train'
    img_dir_small_train = 'train_small'

    img_dir_test_valid = 'valid'
    img_dir_small_test_valid = 'valid_small'
    
    #img_dir2 ="/home/kevin/Programming/Python/ComputerVision/Bird_Classification/bird-species/train_v1/ABBOTTS BABBLER"
    #c = count_items_in_folder(img_dir2)
    #c = select_least_items_folder(img_dir1)

    main(img_dir_train, img_dir_small_train, img_dir_test_valid, img_dir_small_test_valid)


