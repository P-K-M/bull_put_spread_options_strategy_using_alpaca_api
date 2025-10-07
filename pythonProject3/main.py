import zmq

# Initialize ZeroMQ context
context = zmq.Context()

# Create a REQ (Request) socket to communicate with MQL4
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")  # Ensure the port matches the one used in MQL4

# Function to send a request and receive a response
def send_request(request):
    try:
        print(f"Sending request: {request}")
        socket.send_string(request)  # Send the request to the MQL4 server

        # Wait for the reply from the server
        response = socket.recv_string()
        print(f"Received reply: {response}")
        return response
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    # Example request to send "Hello"
    send_request("Hello")

