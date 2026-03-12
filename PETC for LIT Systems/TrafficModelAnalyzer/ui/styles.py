"""
Provides the centralized stylesheets for the application themes, 
ensuring correct rendering of QCheckBox indicators within custom item widgets,
and scaled for 1080p presentation visibility.
"""

DARK_STYLESHEET = """
      QWidget { color: #DDFFFFFF; font-size: 14px; }
      QMainWindow, QWidget { background-color: #000000; }

      QLabel#fieldTitle { color: #AAAAAA; font-size: 15px; }

      QWidget#sidebar, QWidget#statusCard {
          background-color: rgba(255, 255, 255, 0.05);
          border-radius: 8px;
          border: 1px solid rgba(255, 255, 255, 0.1);
      }

      QWidget#graphCard {
          background-color: rgba(255, 255, 255, 0.09);
          border-radius: 8px;
          border: 1px solid rgba(255, 255, 255, 0.05);
      }

      QLineEdit, QComboBox, QTextEdit, QSpinBox, QDoubleSpinBox, QListWidget {
          background-color: #1A1A1A;
          border: 1px solid #444444;
          border-radius: 4px;
          padding: 6px;
          font-size: 14px;
      }
      
      QListWidget::item { 
          padding: 4px; 
      }
      
      QCheckBox::indicator {
          width: 18px;
          height: 18px;
          border: 1px solid #555555;
          border-radius: 3px;
          background-color: #222222;
      }
      
      QCheckBox::indicator:checked {
          background-color: #888888;
          border: 1px solid #AAAAAA;
      }
      
      QPushButton { 
          background-color: #333333; 
          border: 1px solid #555555; 
          padding: 8px; 
          border-radius: 4px; 
          font-size: 14px; 
          font-weight: bold; 
      }
      QPushButton:hover { background-color: #444444; }
      QPushButton:pressed { background-color: #222222; }
      QLabel { background-color: transparent; }

      QTabWidget::pane { border-top: 1px solid #3A3A3A; background-color: transparent; }
      QTabBar::tab { background: transparent; color: #AAAAAA; padding: 12px 24px; border: none; font-family: "Poppins Light"; font-size: 16px; }
      QTabBar::tab:selected { background: #171717; color: #FFFFFF; font-family: "Poppins Medium"; border: none; border-bottom: 2px solid #FFFFFF; }
      
      QHeaderView::section { font-size: 14px; font-weight: bold; padding: 6px; }
      QTableWidget { font-size: 14px; }
"""

LIGHT_STYLESHEET = """
      QWidget { color: #000000; font-size: 14px; }
      QMainWindow, QWidget { background-color: #FFFFFF; }

      QLabel#fieldTitle { color: #555555; font-size: 15px; }

      QWidget#sidebar, QWidget#statusCard {
          background-color: #F8F9FA;
          border-radius: 8px;
          border: 1px solid #DEE2E6;
      }

      QWidget#graphCard {
          background-color: #F8F9FA;
          border-radius: 8px;
          border: 1px solid #DEE2E6;
      }

      QLineEdit, QComboBox, QTextEdit, QSpinBox, QDoubleSpinBox, QListWidget {
          background-color: #FFFFFF;
          border: 1px solid #CCCCCC;
          border-radius: 4px;
          padding: 6px;
          font-size: 14px;
      }
      
      QListWidget::item { 
          padding: 4px; 
      }
      
      QCheckBox::indicator {
          width: 18px;
          height: 18px;
          border: 1px solid #AAAAAA;
          border-radius: 3px;
          background-color: #FFFFFF;
      }
      
      QCheckBox::indicator:checked {
          background-color: #777777;
          border: 1px solid #555555;
      }
      
      QPushButton { 
          background-color: #E1E1E1; 
          border: 1px solid #BDBDBD; 
          padding: 8px; 
          border-radius: 4px; 
          font-size: 14px; 
          font-weight: bold; 
      }
      QPushButton:hover { background-color: #D1D1D1; }
      QPushButton:pressed { background-color: #C1C1C1; }
      QLabel { background-color: transparent; }

      QTabWidget::pane { border-top: 1px solid #CCCCCC; background-color: transparent; }
      QTabBar::tab { background: transparent; color: #777777; padding: 12px 24px; border: none; font-family: "Poppins Light"; font-size: 16px; }
      QTabBar::tab:selected { background: #F0F0F0; color: #000000; font-family: "Poppins Medium"; border: none; border-bottom: 2px solid #000000; }
      
      QHeaderView::section { font-size: 14px; font-weight: bold; padding: 6px; }
      QTableWidget { font-size: 14px; }
"""
