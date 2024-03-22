import os

from plato.load.provide import DatasetProvider
from plato.load.load_pdr import PDRDatasetProvider, RESPONSE_COL2TYPE
from plato.utils.py_utils import make_filename as mkf

SOURCE2SUBTYPE_CATEGORIES = {"cell": ["cancer-type"], "mouse": ["Tumor Type"]}
SOURCE2SUBTYPE_RESPONSE_COL = {"cell": "ln-ic50", "mouse": "min-avg-pct-tumor-growth"}

class PDRSubdatasetProvider(PDRDatasetProvider):
    def __init__(self, source, subtype_category, subtype_name, subtype_response_col, load_from_scratch = False, skip_kg = False, embedding_model = "ComplEx", center_and_standardize = True, cache_dir =None, load_dir=None):

        # Store attributes
        self.source = source
        self.subtype_category = subtype_category
        self.subtype_name = subtype_name
        self.subtype_response_col = subtype_response_col
        self.response_col2type = RESPONSE_COL2TYPE

        self.embedding_model = embedding_model
        self.center_and_standardize = center_and_standardize
        self.cache_file = os.path.join(cache_dir, "{}_{}_{}_{}.pt".format(mkf(self.source), mkf(self.subtype_category), mkf(self.subtype_name), mkf(self.subtype_response_col)))
        print("cache_file: {}".format(self.cache_file))

        self.validate_inputs()

        DatasetProvider.__init__(self, self.embedding_model, self.cache_file, "PDRSubdatasetProvider", load_from_scratch, skip_kg, load_dir=load_dir, cache_dir=cache_dir)

    def validate_inputs(self):
        assert(self.source in ["cell", "mouse"])
        assert(self.subtype_category in SOURCE2SUBTYPE_CATEGORIES[self.source])
        assert(self.subtype_response_col in SOURCE2SUBTYPE_RESPONSE_COL[self.source])
        assert((self.subtype_name in SUBTYPE_CATEGORY2NAMES[self.subtype_category]) or (self.subtype_name == "AllSubtypeNames"))

SUBTYPE_CATEGORY2NAMES = {
"cancer-type": ['Breast Carcinoma', 'Chondrosarcoma', 'Melanoma', 'Non-Small Cell Lung Carcinoma', 'Small Cell Lung Carcinoma'], 
"Tumor Type": ['CRC', 'PDAC', 'BRCA', 'NSCLC', 'CM']}