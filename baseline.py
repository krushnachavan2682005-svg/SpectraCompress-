from split_strategy import RowSplitter
import numpy as np 
from data.data_validator import SchemaValidator # type: ignore
from dataclasses import dataclass
from data.data_contracts import DataContract
schema_validator=SchemaValidator()
@dataclass
class BaselineState:
    top_k_indices: np.ndarray

class TopKVarianceBaseline:
    def __init__(self):
        self.data_contract=DataContract
    def fit_baseline(train_data: np.ndarray, k: int) -> BaselineState :
        variance=np.var(train_data,axis=0) # ithe axis 0 ghetla ahe mhaneje row level std kadycha ahe jr axis specify nahi kela tr tithesaglya data cha ekach varience yeil ani ithe(np.var) var vaparly karan std ani var samechch asta fakt std var cha square root find karta 
        highest_column_indices=np.argsort(variance)
        top_k_indices=highest_column_indices[-k:] # ithe aapn highest ghetly karan jast sread  aslele features mhanje more imformtion ahe tya madhe karan kami information mhanje ek sarkhe features te important nahiye image compress karun pudhe use karyla 
        return BaselineState(top_k_indices=top_k_indices)
    def transform(self,data: np.ndarray, baseline_state: BaselineState) -> np.ndarray:
        schema_validator.validate_schema(data=data,contract=self.data_contract)
        top_k_columns=baseline_state.top_k_indices
        return top_k_columns[:,top_k_columns]


# वर fit_baseline() मध्ये:
# train_data
#    ↓
# variance प्रत्येक column ची
#    ↓
# highest variance columns चे indices
#    ↓
# top_k_indices

# उदा. k=3:

# top_k_indices = [2, 7, 10]

# हे actual data नाहीत, हे फक्त सांगतात:

# "मला column 2, 7 आणि 10 पाहिजेत."

# खाली transform() मध्ये:
# return data[:, top_k_columns]

# म्हणजे:

# new data
#    ↓
# column 2
# column 7
# column 10
#    ↓
# compressed data

# So yes:

# वर → कोणते columns घ्यायचे त्यांचे indices काढले.
# खाली → त्या indices वापरून actual columns/data घेतला. ✅
        
            




