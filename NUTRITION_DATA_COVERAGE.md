# Nutrition data coverage audit

Source: `class_names.json` and `nutrition_table.csv` from the supplied trained-artifact ZIP.

| Model class | Nutrition table mapping | Status |
|---|---|---|
| biryani | — | Missing |
| cholebhature | chole_bhature | Available |
| dabeli | — | Missing |
| dal | — | Missing; dal makhani is not treated as equivalent |
| dhokla | dhokla | Available |
| dosa | masala_dosa | Available; explicit generic-dosa-to-masala-dosa approximation |
| jalebi | jalebi | Available |
| kathiroll | kaathi_rolls | Available |
| kofta | — | Missing |
| naan | butter_naan | Available |
| pakora | pakode | Available |
| paneer | — | Missing; kadai paneer is not treated as generic paneer |
| panipuri | paani_puri | Available |
| pavbhaji | pav_bhaji | Available |
| vadapav | — | Missing |

The application does not calculate or save invented nutrition values for missing classes.
