from fpdf import FPDF

class DoctorVisitPDF(FPDF):
    def __init__():
        
    def header(self):
        self.image('./public/logo_dark.png', 10, 8, 20)
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, "DOCTOR'S VISIT SUMMARY", ln=True, align='C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def section_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_text_color(0)
        self.cell(0, 10, title, ln=True)
        self.set_font('Arial', '', 11)
        self.set_text_color(50)

    def multi_line_list(self, items):
        for item in items:
            self.cell(5)
            self.multi_cell(0, 8, f"- {item}")

# Sample data
