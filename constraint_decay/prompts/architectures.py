CLEAN_ARCHITECTURE_TEMPLATE = """## Architecture

You MUST follow the Clean Architecture pattern. Organize your code into these layers:
- **Routes/Handlers layer**: HTTP request handling, input parsing, response formatting. No business logic.
- **Services/Use Cases layer**: Business logic and orchestration. Framework-agnostic.
- **Models/Entities layer**: Data structures and domain objects.
- **Repository/Data Access layer**: All data storage operations. Accessed only through the Services layer.

Each layer must only depend on the layer below it. Routes import Services, Services import Repositories, never the reverse. Keep each layer in its own directory.
"""