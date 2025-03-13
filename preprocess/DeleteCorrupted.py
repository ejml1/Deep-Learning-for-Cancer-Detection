import os

def remove_files(directory, files):
    for file_name in files:
        file_path = os.path.join(directory, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Removed {file_name}")
        else:
            print(f"{file_name} does not exist")

# Example usage:
if __name__ == "__main__":
    directory = "../BM_cytomorphology_data"
    files_to_remove = [
        "NGB/9001-9968/NGB_09454.jpg",
        "ART/7001-8000/ART_07788.jpg",
        "EBO/6001-7000/EBO_06999.jpg",
        "BLA/9001-10000/BLA_09522.jpg",
        "MYB/5001-6000/MYB_05527.jpg",
        "NGS/4001-5000/NGS_04035.jpg",
        "NGS/18001-19000/NGS_18642.jpg",
        "NGS/15001-16000/NGS_15228.jpg",
        "LYT/19001-20000/LYT_19975.jpg",
        "EOS/2001-3000/EOS_02571.jpg"
    ]
    remove_files(directory, files_to_remove)
