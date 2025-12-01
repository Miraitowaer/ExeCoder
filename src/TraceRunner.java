import com.sun.jdi.*;
import com.sun.jdi.connect.*;
import com.sun.jdi.event.*;
import com.sun.jdi.request.*;

import java.io.IOException;
import java.util.List;
import java.util.Map;

public class TraceRunner {

    public static void main(String[] args) throws IOException, IllegalConnectorArgumentsException, VMStartException, InterruptedException, IncompatibleThreadStateException, AbsentInformationException {
        if (args.length < 1) {
            System.out.println("Usage: java TraceRunner <TargetClassName> [args...]");
            return;
        }
        String targetClass = args[0];

        LaunchingConnector launchingConnector = Bootstrap.virtualMachineManager().defaultConnector();
        Map<String, com.sun.jdi.connect.Connector.Argument> arguments = launchingConnector.defaultArguments();
        arguments.get("main").setValue(targetClass);
        arguments.get("options").setValue("-cp ."); 

        VirtualMachine vm = launchingConnector.launch(arguments);
        vm.eventRequestManager().createClassPrepareRequest().enable();

        EventQueue eventQueue = vm.eventQueue();
        boolean connected = true;

        while (connected) {
            EventSet eventSet = eventQueue.remove();
            for (Event event : eventSet) {
                
                if (event instanceof ClassPrepareEvent) {
                    ClassPrepareEvent classPrepareEvent = (ClassPrepareEvent) event;
                    ReferenceType refType = classPrepareEvent.referenceType();
                    if (refType.name().equals(targetClass)) {
                        MethodEntryRequest methodEntryReq = vm.eventRequestManager().createMethodEntryRequest();
                        methodEntryReq.addClassFilter(targetClass);
                        methodEntryReq.enable();
                    }
                }
                
                if (event instanceof MethodEntryEvent) {
                    MethodEntryEvent entryEvent = (MethodEntryEvent) event;
                    if (entryEvent.method().name().equals("main")) {
                        // 开启 Step Request
                        StepRequest stepRequest = vm.eventRequestManager().createStepRequest(
                                entryEvent.thread(), 
                                StepRequest.STEP_LINE, 
                                StepRequest.STEP_INTO // <--- 改为 INTO，深入函数内部
                        );
                        stepRequest.addClassFilter(targetClass); 
                        stepRequest.enable();
                    }
                }

                if (event instanceof StepEvent) {
                    printTrace((StepEvent) event);
                }

                if (event instanceof VMDisconnectEvent) {
                    connected = false;
                }
            }
            eventSet.resume();
        }
    }

    private static void printTrace(LocatableEvent event) {
        try {
            StackFrame frame = event.thread().frame(0);
            Location location = frame.location();
            int line = location.lineNumber();
            
            System.out.print("{\"line\": " + line + ", \"vars\": {");
            
            List<LocalVariable> visibleVariables = frame.visibleVariables();
            boolean first = true;
            for (LocalVariable var : visibleVariables) {
                Value value = frame.getValue(var);
                if (!first) System.out.print(", ");
                String valStr = (value != null) ? value.toString().replace("\"", "\\\"") : "null";
                System.out.print("\"" + var.name() + "\": \"" + valStr + "\"");
                first = false;
            }
            System.out.println("}}");
            
        } catch (Exception e) { }
    }
}