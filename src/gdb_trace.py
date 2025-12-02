
import gdb, json, os
class TraceCommand(gdb.Command):
    def __init__(self): super(TraceCommand, self).__init__("trace_run", gdb.COMMAND_USER)
    def invoke(self, arg, from_tty):
        gdb.execute("set pagination off"); gdb.execute("set confirm off")
        try: target = gdb.parse_and_eval("SOURCE_FILE").string(); target=os.path.basename(target)
        except: target = None
        try: gdb.execute("start", to_string=True)
        except: return
        while True:
            try:
                f = gdb.newest_frame(); sal = f.find_sal()
                if sal.line > 0:
                    valid = True
                    if target and sal.symtab and sal.symtab.filename:
                        if os.path.basename(sal.symtab.filename) != target: valid = False
                    if valid:
                        vs = {}
                        try:
                            b = f.block()
                            while b:
                                if not b.is_global:
                                    for s in b:
                                        if (s.is_variable or s.is_argument) and s.name not in vs:
                                            try: 
                                                val = str(s.value(f))
                                                if "=" in val: val = val.split("=")[-1].strip() # Clean C++ output
                                                vs[s.name] = val.split()[0] 
                                            except: pass
                                b = b.superblock
                        except: pass
                        print("JSON_TRACE: " + json.dumps({"line": sal.line, "vars": vs}))
                gdb.execute("step", to_string=True)
            except: break
        gdb.execute("quit")
TraceCommand()
