"""Phase 4c finding extraction: taxonomy EntityRuler + negspaCy negation.

Replaces the crude regex+fixed-window labeler (`graph.dataset._label_text`) with proper
sentence-scoped clinical negation. We have a *closed* finding taxonomy, so instead of
generic biomedical NER we drive a spaCy EntityRuler with curated synonyms per finding,
then let negspaCy's NegEx mark negated mentions. A finding is positive iff it has a
non-negated mention. Lightweight: a blank English pipeline (no model download) +
sentencizer + entity_ruler + negex.

    nlp = build_nlp()
    extract("No ascites. Hepatic steatosis present.", nlp)
    -> {"ascites": 0, "hepatic_steatosis": 1, ...}
"""
from __future__ import annotations

# finding -> surface forms (case-insensitive phrase match). Extends the 6 GNN findings
# with the atomic findings the Phase 4 rules consume (wall thickening, fluid, dilation...).
SYNONYMS: dict[str, list[str]] = {
    "hepatic_steatosis": ["hepatic steatosis", "fatty liver", "fatty infiltration",
                          "steatosis", "hepatic steatofibrosis", "diffuse fatty infiltration"],
    "gallstones": ["gallstone", "gallstones", "cholelithiasis", "cholelith",
                   "calculous", "gallbladder stone", "gallbladder stones"],
    "splenomegaly": ["splenomegaly", "enlarged spleen", "spleen is enlarged",
                     "splenic enlargement"],
    "ascites": ["ascites", "ascitic fluid", "intraperitoneal fluid", "peritoneal fluid"],
    "aortic_aneurysm": ["aneurysm", "aortic aneurysm", "aneurysmal"],
    "hepatic_lesion": ["hepatic lesion", "liver lesion", "hepatic cyst", "liver cyst",
                       "hepatic mass", "liver mass", "hepatic metastasis", "liver metastasis",
                       "hepatic metastases", "liver metastases", "hepatic nodule"],
    "liver_nodularity": ["nodular liver", "liver nodularity", "nodular contour",
                         "nodular hepatic contour", "cirrhotic morphology"],
    "cirrhosis": ["cirrhosis", "cirrhotic"],
    "varices": ["varices", "varix", "variceal"],
    "gallbladder_wall_thickening": ["gallbladder wall thickening", "gb wall thickening",
                                    "thickened gallbladder wall"],
    "pericholecystic_fluid": ["pericholecystic fluid"],
    "cholecystitis": ["cholecystitis"],
    "pancreatitis": ["pancreatitis", "peripancreatic stranding", "pancreatic stranding"],
    "peripancreatic_fluid": ["peripancreatic fluid"],
    "hydronephrosis": ["hydronephrosis", "hydroureteronephrosis", "pelvicaliectasis",
                       "collecting system dilation", "dilated collecting system"],
    "renal_calculus": ["nephrolithiasis", "renal calculus", "renal calculi", "kidney stone",
                       "ureterolithiasis", "ureteral calculus", "obstructing stone"],
    "renal_cyst": ["renal cyst", "kidney cyst"],
    "adrenal_nodule": ["adrenal nodule", "adrenal adenoma", "adrenal mass"],
    "bowel_obstruction": ["bowel obstruction", "obstruction", "dilated loops of bowel",
                          "dilated small bowel", "transition point", "sbo"],
    "bowel_wall_thickening": ["bowel wall thickening", "wall thickening"],
    "pneumoperitoneum": ["pneumoperitoneum", "free air", "free intraperitoneal air",
                         "extraluminal air"],
    "lymphadenopathy": ["lymphadenopathy", "enlarged lymph nodes", "enlarged lymph node"],
    # --- additional taxonomy (extension, ~28 new findings) -------------------------
    # Pancreas
    "pancreatic_mass": ["pancreatic mass", "pancreatic tumor", "pancreatic tumour",
                        "pancreatic carcinoma", "pancreatic adenocarcinoma",
                        "ductal adenocarcinoma", "IPMN",
                        "intraductal papillary mucinous neoplasm",
                        "pancreatic neuroendocrine tumor",
                        "pancreatic cystic neoplasm", "mucinous cystic neoplasm",
                        "serous cystadenoma"],
    "pancreatic_duct_dilation": ["pancreatic duct dilation", "pancreatic duct dilatation",
                                 "dilated pancreatic duct", "main pancreatic duct dilatation"],
    "pancreatic_atrophy": ["pancreatic atrophy", "atrophic pancreas"],
    # Liver lesions (split out from generic hepatic_lesion)
    "hepatic_cyst": ["hepatic cyst", "hepatic cysts", "liver cyst", "liver cysts"],
    "hepatic_hemangioma": ["hepatic hemangioma", "liver hemangioma",
                           "cavernous hemangioma"],
    "hepatic_abscess": ["hepatic abscess", "liver abscess", "pyogenic abscess"],
    "hepatocellular_carcinoma": ["hepatocellular carcinoma", "HCC", "hepatoma"],
    "hepatic_metastasis": ["hepatic metastasis", "hepatic metastases",
                           "liver metastasis", "liver metastases", "hepatic mets",
                           "metastatic disease in the liver"],
    # Biliary
    "bile_duct_dilation": ["intrahepatic biliary ductal dilation",
                           "intrahepatic biliary ductal dilatation",
                           "extrahepatic biliary ductal dilation",
                           "extrahepatic biliary ductal dilatation",
                           "biliary ductal dilation", "biliary ductal dilatation",
                           "common bile duct dilation", "CBD dilation", "dilated CBD"],
    "pneumobilia": ["pneumobilia", "biliary air", "air in the biliary tree"],
    "gallbladder_polyp": ["gallbladder polyp", "gallbladder polyps", "GB polyp",
                          "polyp in the gallbladder"],
    # Spleen
    "splenic_infarct": ["splenic infarct", "splenic infarction"],
    "splenic_laceration": ["splenic laceration", "spleen laceration"],
    # Kidney / urinary
    "renal_mass": ["renal mass", "renal masses", "kidney mass", "kidney masses",
                   "renal cell carcinoma", "RCC", "renal tumor", "renal tumour"],
    "pyelonephritis": ["pyelonephritis"],
    "bladder_mass": ["bladder mass", "bladder tumor", "bladder carcinoma",
                     "urinary bladder mass"],
    # Adrenal
    "adrenal_mass": ["adrenal mass", "adrenal tumor", "adrenal carcinoma"],
    # Bowel / appendix
    "appendicitis": ["appendicitis", "inflamed appendix"],
    "diverticulitis": ["diverticulitis"],
    "colitis": ["colitis"],
    # Vascular
    "aortic_dissection": ["aortic dissection", "dissection of the aorta",
                          "type a dissection", "type b dissection", "intimal flap"],
    "aortic_atherosclerosis": ["aortic atherosclerosis", "atherosclerotic aorta",
                               "aortic calcification", "calcified aorta",
                               "aortic mural calcification"],
    "portal_vein_thrombosis": ["portal vein thrombosis", "portal vein thrombus",
                               "portal venous thrombosis"],
    "ivc_thrombus": ["IVC thrombus", "inferior vena cava thrombosis",
                     "inferior vena cava thrombus"],
    # Lower thorax / visible at abdomen CT
    "pleural_effusion": ["pleural effusion", "pleural effusions"],
    "pulmonary_nodule": ["pulmonary nodule", "pulmonary nodules", "lung nodule",
                         "lung nodules"],
    # MSK
    "vertebral_fracture": ["vertebral fracture", "compression fracture",
                           "vertebral compression fracture", "vertebral body fracture"],
    # Retroperitoneal
    "retroperitoneal_hematoma": ["retroperitoneal hematoma", "retroperitoneal hemorrhage",
                                 "retroperitoneal bleed"],
}
FINDING_NAMES = list(SYNONYMS)


def build_nlp():
    """Blank English pipeline + sentencizer + taxonomy EntityRuler + negspaCy NegEx."""
    import spacy
    from negspacy.negation import Negex  # noqa: F401 — registers the "negex" factory
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    ruler = nlp.add_pipe("entity_ruler", config={"phrase_matcher_attr": "LOWER",
                                                 "validate": False})
    ruler.add_patterns([{"label": f, "pattern": p}
                        for f, forms in SYNONYMS.items() for p in forms])
    nlp.add_pipe("negex", config={"ent_types": FINDING_NAMES})
    return nlp


def extract(text: str, nlp) -> dict[str, int]:
    """{finding: 1 if a non-negated mention exists else 0}."""
    out = {f: 0 for f in FINDING_NAMES}
    doc = nlp(str(text))
    for ent in doc.ents:
        if ent.label_ in out and not ent._.negex:
            out[ent.label_] = 1
    return out
