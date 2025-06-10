import os
import pandas as pd

def buildravdess(ravdess_path):
    emotion_map = {
        1: 'neutral',
        2: 'calm',
        3: 'happy',
        4: 'sad',
        5: 'angry',
        6: 'fear',
        7: 'disgust',
        8: 'surprise'
    }

    desired_emotions = {'neutral', 'happy', 'sad', 'angry'}
    data = []

    for actor_folder in sorted(os.listdir(ravdess_path)):
        actor_path = os.path.join(ravdess_path, actor_folder)
        if not os.path.isdir(actor_path):
            continue
        actor_id = int(actor_folder.split('_')[1])

        for filename in sorted(os.listdir(actor_path)):
            if filename.endswith('.wav'):
                parts = filename.split('-')
                emotion_code = int(parts[2])
                emotion = emotion_map[emotion_code]
                if emotion not in desired_emotions:
                    continue
                full_path = os.path.join(ravdess_path, actor_folder, filename)
                data.append([full_path, emotion, actor_id])

    return pd.DataFrame(data, columns=["relative_path", "emotion", "actor_id"])
