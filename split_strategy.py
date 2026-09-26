
  # logic asa ahe ki data madhun train ani validation chya rows baher kadychya ahet so jevdya validation ratio dila ahe tevdya rows baher kadychya aeht ani baher kadtana 

import numpy as np

class RowSplitter:

    @staticmethod
    def safe_split(
        data: np.ndarray,
        val_ratio: float,
        seed: int
    ) -> tuple[np.ndarray, np.ndarray]:

        total_rows = len(data)

        validation_rows = int(val_ratio * total_rows)

        rng = np.random.default_rng(seed)

        indices = np.arange(total_rows)  ## sagla data shuffel hoto tyamule tithe training ani validation la proper data jato 
        rng.shuffle(indices)
        val_indices=indices[-validation_rows:]# hyacha meaning asa ahe ki  last kadun jevdhe elements ahet tevde sagle 
        train_indices=indices[:-validation_rows] # hyacha meaning asa aehe last chec element sodun sagle element ghe 
        return(train_indices,val_indices)


        

