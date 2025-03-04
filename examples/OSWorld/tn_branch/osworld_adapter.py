from osworld import OSWorld  

class OSWorldAdapter:
    def __init__(self):
        self.osworld = OSWorld()

    def execute_command(self, command):
        try:
            result = self.osworld.run(command)
            return result
        except Exception as e:
            return f"Error: {str(e)}"

    def perform_action(self, action_description):
        if "create file" in action_description:
            filename = action_description.split("create file")[1].strip()
            return self.execute_command(f"touch {filename}")
        elif "list directory" in action_description:
            return self.execute_command("ls")
        else:
            return "Unsupported action"
