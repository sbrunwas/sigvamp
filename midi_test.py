import mido

inputs = mido.get_input_names()

print("Available MIDI inputs:")
for i, name in enumerate(inputs):
    print(i, name)

port_number = int(input("Select port number: "))

with mido.open_input(inputs[port_number]) as port:
    print(f"Listening on: {inputs[port_number]}")

    for msg in port:
        print(msg)
