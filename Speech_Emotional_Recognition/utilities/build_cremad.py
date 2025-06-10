import os
import pandas as pd

def buildcremad(cremad_path):
    emotion_map = {
        'ANG': 'angry',
        'DIS': 'disgust',
        'FEA': 'fear',
        'HAP': 'happy',
        'NEU': 'neutral',
        'SAD': 'sad'
    }

    desired_emotions = {'neutral', 'happy', 'sad', 'angry'}
    data = []

    for filename in sorted(os.listdir(cremad_path)):
        if filename.endswith('.wav'):
            parts = filename.split('_')
            emotion_code = parts[2]
            emotion = emotion_map.get(emotion_code)

            if emotion not in desired_emotions:
                continue

            actor_id = int(parts[0][-3:])
            full_path = os.path.join(cremad_path, filename)
            data.append([full_path, emotion, actor_id])

    return pd.DataFrame(data, columns=["relative_path", "emotion", "actor_id"])
