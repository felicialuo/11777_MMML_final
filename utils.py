import pathlib

def get_labels(root_path):
    dataset_root_path = pathlib.Path(root_path)

    video_count_train = len(list(dataset_root_path.glob('train/*/*.avi')))
    video_count_test_seen = len(list(dataset_root_path.glob('test_seen/*/*.avi')))
    video_count_test_unseen = len(list(dataset_root_path.glob('test_unseen/*/*.avi')))

    all_video_file_paths = (
        list(dataset_root_path.glob("train/*/*.avi"))
        +
        list(dataset_root_path.glob("test_seen/*/*.avi"))
        +
        list(dataset_root_path.glob("test_unseen/*/*.avi"))
    )


    class_labels = sorted({str(path).split("\\")[-2] for path in all_video_file_paths})
    label2id = {label: i for i, label in enumerate(class_labels)}
    id2label = {i: label for label, i in label2id.items()}

    return label2id, id2label

