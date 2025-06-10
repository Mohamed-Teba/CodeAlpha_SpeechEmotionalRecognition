import os
import pandas as pd

def buildtess(tess_path):
    emotion_map = {
        'angry': 'angry',
        'disgust': 'disgust',
        'fear': 'fear',
        'happy': 'happy',
        'neutral': 'neutral',
        'ps': 'happy',
        'sad': 'sad'
    }

    desired_emotions = {'neutral', 'happy', 'sad', 'angry'}
    data = []

    for root, _, files in os.walk(tess_path):
        for file in sorted(files):
            if file.endswith('.wav'):
                for key in emotion_map:
                    if f'_{key.upper()}' in file.upper():
                        emotion = emotion_map[key]
                        if emotion not in desired_emotions:
                            break
                        actor_id = 1 if 'YAF' in file else 2
                        full_path = os.path.join(root, file)
                        data.append([full_path, emotion, actor_id])
                        break

    return pd.DataFrame(data, columns=["relative_path", "emotion", "actor_id"])
