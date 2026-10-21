"""
corpus.py - the documents the search engine indexes.

* load_newsgroups()  : real news posts from the 20 Newsgroups dataset (6 topics).
                       scikit-learn downloads it automatically on first use (needs internet).
* SAMPLE_ROWS        : 36 small built-in documents, so the app ALWAYS works offline.
* EVAL_QUERIES       : (query, relevant category) pairs used by the Evaluation tab.
"""

CATEGORIES = [
    "sci.space",
    "rec.sport.hockey",
    "comp.graphics",
    "sci.med",
    "rec.autos",
    "sci.crypt",
]

_SAMPLE = {
    "sci.space": [
        ("NASA schedules shuttle launch",
         "NASA has scheduled the next space shuttle launch for early spring. The orbiter will carry a crew of seven astronauts and a satellite payload into low Earth orbit."),
        ("Mars rover finds evidence of ancient water",
         "The Mars rover sent back images of rock layers that suggest liquid water once flowed on the planet. Scientists hope the samples will reveal whether microbial life ever existed there."),
        ("Telescope discovers new planets",
         "A space telescope has detected several planets orbiting distant stars. Astronomers measure the dip in starlight as a planet passes in front of its star."),
        ("Rocket engine test succeeds",
         "Engineers completed a static fire test of a new rocket engine designed to lift heavy cargo into orbit. The booster will be reusable and land back on Earth after launch."),
        ("Moon base proposal",
         "A proposal describes a permanent lunar base near the south pole of the Moon where ice could provide water and fuel for astronauts."),
        ("Space station crew returns",
         "The crew of the orbital space station returned to Earth after six months in microgravity, studying how long missions affect the human body."),
    ],
    "rec.sport.hockey": [
        ("Playoff series goes to game seven",
         "The hockey playoff series between the two rivals will be decided in game seven. The goalie made forty saves as the team forced overtime in the previous game."),
        ("Star center signs contract",
         "The star center signed a five year contract extension with the team. He scored fifty goals last season and led the league in assists."),
        ("Coach changes defensive strategy",
         "The coach adjusted the defensive strategy after the team allowed too many shots. The power play unit also struggled during the road trip."),
        ("Rookie scores hat trick",
         "A rookie winger scored a hat trick in his home debut as the team beat its division rival. Fans threw hats onto the ice to celebrate."),
        ("League announces expansion",
         "The hockey league announced a new expansion franchise that will begin play next season. The arena will seat eighteen thousand fans for every game."),
        ("Injury hits the blue line",
         "The team lost its top defenseman to a shoulder injury just before the playoffs. The trainer said he will miss at least four weeks of the season."),
    ],
    "comp.graphics": [
        ("Ray tracing renders realistic images",
         "Ray tracing software renders realistic images by simulating the path of light through a 3D scene. Reflections, shadows and refraction are computed for each pixel."),
        ("Choosing an image file format",
         "The JPEG format compresses photographs with some loss of quality while the PNG format keeps every pixel exact. GIF files support simple animation and a limited color palette."),
        ("Polygon mesh rendering",
         "The graphics library draws a 3D model as a mesh of polygons and applies texture mapping to give surfaces detail. The renderer uses a depth buffer to hide surfaces behind other objects."),
        ("Graphics card driver update",
         "A new graphics card driver improves frame rates and fixes screen flicker in several programs. Users can download the update from the vendor website."),
        ("Image processing filters",
         "Image processing programs apply filters such as blur, sharpen and edge detection by convolving each pixel with its neighbors. Color images are handled channel by channel."),
        ("Vector versus bitmap graphics",
         "Vector graphics describe shapes with curves and scale to any size without losing sharpness, while bitmap graphics store a grid of colored pixels and become blurry when enlarged."),
    ],
    "sci.med": [
        ("New treatment for high blood pressure",
         "Doctors report that a new medication lowers blood pressure in patients who did not respond to older drugs. Side effects were mild and included headache and dizziness."),
        ("Vaccine trial shows strong immune response",
         "A clinical trial found that the vaccine produced a strong immune response and protected patients against infection. Researchers will now study how long the protection lasts."),
        ("Diet and heart disease",
         "A diet low in saturated fat and rich in vegetables can reduce the risk of heart disease. Physicians recommend regular exercise and regular checkups with a doctor."),
        ("Diagnosing diabetes early",
         "Early diagnosis of diabetes depends on blood glucose tests. Untreated disease can damage the kidneys, eyes and nerves, so patients should follow their treatment plan."),
        ("Antibiotic resistance warning",
         "Hospitals warn that overuse of antibiotics is creating resistant bacteria. Doctors should prescribe the drugs only when an infection is confirmed."),
        ("Sleep and health",
         "Chronic lack of sleep is linked to obesity, weakened immunity and depression. Medical experts advise adults to sleep seven to nine hours each night."),
    ],
    "rec.autos": [
        ("Choosing a reliable used car",
         "When buying a used car, check the maintenance records, test drive it, and have a mechanic inspect the engine and transmission before agreeing on a price with the dealer."),
        ("Hybrid engines improve fuel economy",
         "Hybrid cars combine a gasoline engine with an electric motor to improve fuel economy in city driving. The battery recharges while braking."),
        ("Brake pad replacement",
         "Squealing brakes usually mean the brake pads are worn. Replacing pads and resurfacing the rotors restores stopping power and keeps the driver safe."),
        ("Sports car handling review",
         "The new sports car has a powerful engine, precise steering and a stiff suspension that grips corners at high speed. The dealer quotes a price near forty thousand dollars."),
        ("Winter tires and driving safety",
         "Winter tires give better traction on snow and ice. Drivers should also check the battery, wiper blades and antifreeze before the cold season."),
        ("Oil change intervals",
         "Most manufacturers recommend changing engine oil every five thousand miles. Fresh oil reduces wear and helps the engine run longer."),
    ],
    "sci.crypt": [
        ("How public key encryption works",
         "Public key encryption uses a pair of keys: a public key to encrypt messages and a private key to decrypt them. Only the owner of the private key can read the message."),
        ("Hash functions and passwords",
         "A cryptographic hash function turns a password into a fixed length digest. Secure systems add a random salt so identical passwords produce different hashes."),
        ("Key length and brute force attacks",
         "A longer key makes brute force attacks impractical because the number of possible keys grows exponentially. Modern ciphers use keys of at least one hundred twenty eight bits."),
        ("Digital signatures verify authors",
         "A digital signature lets a receiver verify that a message came from the claimed sender and was not altered. The sender signs a hash of the message with a private key."),
        ("Government debates encryption policy",
         "Lawmakers debate whether security agencies should be able to read encrypted communication. Privacy advocates argue that weakening encryption makes everyone less secure."),
        ("Symmetric ciphers and block modes",
         "Symmetric ciphers such as AES encrypt data in blocks using a shared secret key. The mode of operation determines how blocks are chained together and whether patterns leak."),
    ],
}

SAMPLE_ROWS = [
    {"title": title, "text": text, "category": category}
    for category, docs in _SAMPLE.items()
    for title, text in docs
]

EVAL_QUERIES = [
    ("space shuttle orbit launch astronauts", "sci.space"),
    ("planets and telescope observations of distant stars", "sci.space"),
    ("hockey playoff goalie team game", "rec.sport.hockey"),
    ("player contract goals season league", "rec.sport.hockey"),
    ("image rendering 3D graphics pixels", "comp.graphics"),
    ("file format compression images color", "comp.graphics"),
    ("doctor treatment patients disease medication", "sci.med"),
    ("health risk diet infection", "sci.med"),
    ("car engine driving dealer price", "rec.autos"),
    ("tires brakes maintenance safety", "rec.autos"),
    ("encryption private public key message", "sci.crypt"),
    ("password hash security attack", "sci.crypt"),
]


def load_newsgroups(per_category=200):
    """Download (first run only) and return up to `per_category` posts per topic."""
    from sklearn.datasets import fetch_20newsgroups

    data = fetch_20newsgroups(
        subset="train",
        categories=CATEGORIES,
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=42,
    )
    counts = {name: 0 for name in data.target_names}
    rows = []
    for text, target in zip(data.data, data.target):
        category = data.target_names[target]
        text = " ".join(text.split())
        if len(text) < 80 or counts[category] >= per_category:
            continue
        counts[category] += 1
        rows.append(
            {
                "title": text[:70] + ("..." if len(text) > 70 else ""),
                "text": text,
                "category": category,
            }
        )
    return rows