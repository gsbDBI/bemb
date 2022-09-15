import os
import numpy as np
import pandas as pd


if __name__ == '__main__':
    psid_all_transitions = pd.read_csv(
        '/home/tianyudu/Development/PSID/psid_public_dataset.csv',
        low_memory=False)
    output_path = './psid_by_decade/'
    # Year_1 values are 2-digit.
    psid_all_transitions['Decade_1'] = (
        psid_all_transitions['Year_1'] // 10 * 10)
    for decade in psid_all_transitions['Decade_1'].unique():
        psid_decade_transitions = psid_all_transitions[psid_all_transitions['Decade_1'] == decade]
        print(f'psid_{decade}s', len(psid_decade_transitions), 'transitions')
        psid_decade_transitions.to_csv(
            os.path.join(
                output_path,
                f'psid_{decade}s'),
            index=False)
