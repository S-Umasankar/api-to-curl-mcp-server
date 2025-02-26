sdk = MCPSDK()
print(sdk.generate_dataset())    # Triggers dataset generation
print(sdk.preprocess_data())     # Preprocesses data
print(sdk.train_model())         # Starts training
print(sdk.generate_curl("GET /users/{id} Retrieves user details by ID."))  # Converts API to cURL
