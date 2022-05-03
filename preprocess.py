import pandas as pd
import pickle
import os
from tqdm import tqdm, trange
from os import listdir
from os.path import isfile, join

def load_data(folder: str, file: str) -> pd.DataFrame:
    df = pd.read_csv(folder + os.sep + file, sep='\t')
    return df

def group_event(df: pd.DataFrame) -> dict:
    visits_by_visitors = {}
    for _, row in enumerate(tqdm(df.iterrows())):
        visitorid, itemid, timestamp = row[1].values

        if visitorid not in visits_by_visitors:
            visits_by_visitors[visitorid] = {'itemids': [], 'timestamps': []}
        visits_by_visitors[visitorid]['itemids'].append(itemid)
        visits_by_visitors[visitorid]['timestamps'].append(timestamp)
    return visits_by_visitors

def create_sessions(visits_by_visitors: dict, session_duration=7200000) -> dict:
    # Let's group events from visitors into sessions.
    sessions_by_visitors = {}
    for visitorid, visitor_dict in visits_by_visitors.items():
        sessions = [[]]
        events_sorted = sorted(zip(visitor_dict['timestamps'],
                                   visitor_dict['itemids']))
        for i in range(len(events_sorted) - 1):
            sessions[-1].append(events_sorted[i][1])
            if (events_sorted[i + 1][0] - events_sorted[i][0]) > session_duration:
                sessions.append([])
        sessions[-1].append(events_sorted[len(events_sorted) - 1][1])
        sessions_by_visitors[visitorid] = sessions
        return sessions_by_visitors, sessions

def extract_subsessions(sessions):
    """Extracts all partial sessions from the sessions given.

    For example, a session (1, 2, 3) should be augemnted to produce two
    separate sessions (1, 2) and (1, 2, 3).
    """
    all_sessions = []
    for session in sessions:
        for i in range(1, len(session)):
            all_sessions.append(session[:i+1])
    return all_sessions

def create_sets(sessions_by_visitors: dict) -> list:
    out_session = []
    all_visitors = list(sessions_by_visitors.keys())

    for visitor in all_visitors:
        out_session.extend(extract_subsessions(sessions_by_visitors[visitor]))
    return out_session


if __name__ == "__main__":

    files = [f for f in listdir('./raw') if 'item_views_' in f and isfile(join('./raw', f))]
    print('check')
    for file in files:
        df = load_data('raw', file)
        visits_by_visitors = group_event(df)

        sessions_by_visitors, sessions = create_sessions(visits_by_visitors)

        all_sessions = extract_subsessions(sessions)

        out_session = create_sets(sessions_by_visitors)

        # Save the processed files.
        with open(f'processed/{file}', 'wb') as f:
            pickle.dump(out_session, f)
