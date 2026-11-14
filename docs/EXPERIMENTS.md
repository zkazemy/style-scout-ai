1-
 uv run python -c "
import sqlite3
from stylescout.catalog.embeddings import query_index
db = sqlite3.connect('data/catalog.db')
for q in ['black sneakers', 'something for a summer wedding', 'cheap casual shirt']:
    print('\n==', q)
    for pid, dist in query_index(q, n_results=5):
        name, cat, color, price = db.execute(
            'select name, category, color, price_eur from products where product_id=?', (pid,)).fetchone()
        print(f'  {dist:.3f} | {name} | {cat} | {color} | {price} EUR')
"

== black sneakers
  0.242 | Converse Unisex Cyclone Black Canvas Shoe | shoes | black | 70.43 EUR
  0.283 | Flying Machine Men Black Casual Shoes | shoes | black | 43.02 EUR
  0.284 | Skechers Men Black Sneakers | shoes | black | 181.04 EUR
  0.292 | Bata Men Macho Black Sandals | shoes | black | 179.67 EUR
  0.293 | Converse Unisex Chuck Logo Print Black Canvas | shoes | black | 157.0 EUR

== something for a summer wedding
  0.603 | FNF Orange Wedding Collection Sari | dress | orange | 138.27 EUR
  0.617 | Enamor Women Navy Blue Baby Doll Nightdress | dress | navy blue | 89.44 EUR
  0.620 | Palm Tree Kids Girl Washed Blue Skirts | dress | blue | 100.02 EUR
  0.622 | Palm Tree Kids Girls Red Saloppette Set | dress | red | 83.89 EUR
  0.627 | FNF Green & Purple Wedding Collection Sari | dress | green | 112.24 EUR

== cheap casual shirt
  0.378 | Proline Grey Polo T-shirt | shirt | grey | 28.2 EUR
  0.391 | Guerrilla Men's Wanted White T-shirt | shirt | white | 36.38 EUR
  0.406 | W Women's Deep U Neck Short Sleeve Black Top | shirt | black | 108.03 EUR
  0.409 | Probase Blue & Black Checked Slim Fit Casual Shirt | shirt | blue | 39.92 EUR
  0.412 | Jealous 21 Green Bee Mine T-shirt | shirt | green | 115.98 EUR



  2- 