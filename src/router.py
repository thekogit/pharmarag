import re

class RegulatoryRouter:
    def __init__(self):
        self.patterns = {
            "INDICATIONS": r"(?i)^\d+\s+INDICATIONS\s+AND\s+USAGE",
            "DOSAGE": r"(?i)^\d+\s+DOSAGE\s+AND\s+ADMINISTRATION",
            "CONTRAINDICATIONS": r"(?i)^\d+\s+CONTRAINDICATIONS",
            "WARNINGS": r"(?i)^\d+\s+WARNINGS\s+AND\s+PRECAUTIONS",
            "ADVERSE_REACTIONS": r"(?i)^\d+\s+ADVERSE\s+REACTIONS",
            "CLINICAL_STUDIES": r"(?i)^\d+\s+CLINICAL\s+STUDIES"
        }

    def parse(self, text: str) -> dict[str, str]:
        sections = {"UNCLASSIFIED": ""}
        current_section = "UNCLASSIFIED"
        
        lines = text.splitlines()
        for line in lines:
            found_header = False
            for section_name, pattern in self.patterns.items():
                if re.match(pattern, line.strip()):
                    current_section = section_name
                    if current_section not in sections:
                        sections[current_section] = ""
                    found_header = True
                    break
            
            if not found_header:
                if sections[current_section]:
                    sections[current_section] += "\n" + line
                else:
                    sections[current_section] += line
                    
        # Cleanup: remove empty sections
        return {k: v.strip() for k, v in sections.items() if v.strip()}
