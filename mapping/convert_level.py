from PIL import Image

# --- Config ---
TILE_SIZE = 16
TILESET_PATH = "NES - Super Mario Bros - Tileset.png"
ENEMIES_PATH = "NES - Super Mario Bros - Enemies & Bosses.png"
LEVEL_PATH = "gan_level_20250517-214855.txt"
OUTPUT_PATH = "level_output.png"

# --- Define mapping: character -> (x, y) of tile in spritesheet ---
tile_pixel_map = {
    "X": (0, 16),                       # Ground block
    "S": (16, 16),                      # Mid-air tile brick
    "]": (136, 213)    # Right side of green pipe
}

enemies_pixel_map = {
    "E": (0, 16)  # Enemy mushroom
}

# --- Load level text ---
with open(LEVEL_PATH, "r") as f:
    level = [line.rstrip('\n') for line in f]

width, height = len(level[0]), len(level)
output_img = Image.new("RGBA", (width * TILE_SIZE, height * TILE_SIZE))
tileset_img = Image.open(TILESET_PATH)
enemies_img = Image.open(ENEMIES_PATH)

# --- Build level image ---
for y, row in enumerate(level):
    for x, ch in enumerate(row):
        # if ch not in tile_pixel_map:
        #     output_img.paste((92, 148, 252, 255), (x * TILE_SIZE, y * TILE_SIZE, (x + 1) * TILE_SIZE, (y + 1) * TILE_SIZE))
        #     continue
        # sx, sy = tile_pixel_map[ch]
        # tile = tileset_img.crop((sx, sy, sx + TILE_SIZE, sy + TILE_SIZE))
        # output_img.paste(tile, (x * TILE_SIZE, y * TILE_SIZE))
        tile_pos = tile_pixel_map.get(ch)
        enemy_pos = enemies_pixel_map.get(ch)

        if tile_pos:
            sx, sy = tile_pos
            tile = tileset_img.crop((sx, sy, sx + TILE_SIZE, sy + TILE_SIZE))
            output_img.paste(tile, (x * TILE_SIZE, y * TILE_SIZE))

        elif enemy_pos:
            sx, sy = enemy_pos
            tile = enemies_img.crop((sx, sy, sx + TILE_SIZE, sy + TILE_SIZE))
            output_img.paste(tile, (x * TILE_SIZE, y * TILE_SIZE))

        else:
            # Fill with sky color
            output_img.paste((148, 148, 252, 255), (x * TILE_SIZE, y * TILE_SIZE, (x + 1) * TILE_SIZE, (y + 1) * TILE_SIZE))

output_img.save(OUTPUT_PATH)
print(f"Level saved as {OUTPUT_PATH}")
