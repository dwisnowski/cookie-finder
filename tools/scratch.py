import gpiod

chip = gpiod.Chip("/dev/gpiochip1")

lines = chip.request_lines(
    consumer="stepper",
    config={
        258: gpiod.LineSettings(direction=gpiod.line.Direction.OUTPUT),
        268: gpiod.LineSettings(direction=gpiod.line.Direction.OUTPUT),
        271: gpiod.LineSettings(direction=gpiod.line.Direction.OUTPUT),
        272: gpiod.LineSettings(direction=gpiod.line.Direction.OUTPUT),
    },
)

lines.set_values({
    258: gpiod.line.Value.ACTIVE,
    268: gpiod.line.Value.INACTIVE,
    271: gpiod.line.Value.INACTIVE,
    272: gpiod.line.Value.INACTIVE,
})

print("STEP OK")
seq = [
    (1,0,0,0),
    (0,1,0,0),
    (0,0,1,0),
    (0,0,0,1),
]

try:
    while True:
        for step in seq:
            lines.set_values({
                258: gpiod.line.Value.ACTIVE if step[0] else gpiod.line.Value.INACTIVE,
                268: gpiod.line.Value.ACTIVE if step[1] else gpiod.line.Value.INACTIVE,
                271: gpiod.line.Value.ACTIVE if step[2] else gpiod.line.Value.INACTIVE,
                272: gpiod.line.Value.ACTIVE if step[3] else gpiod.line.Value.INACTIVE,
            })
            time.sleep(0.1)
finally:
    lines.release()