from llm_reasoners import OSWorldReasoner

def main():
    reasoner = OSWorldReasoner()
    task = "Create a file named 'test.txt' and list the directory contents."
    results = reasoner.reason(task)
    for result in results:
        print(result)

if __name__ == "__main__":
    main()
