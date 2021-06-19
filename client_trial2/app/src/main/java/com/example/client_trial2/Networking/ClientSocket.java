package com.example.client_trial2.Networking;

import com.example.client_trial2.Networking.Enums.BufferState;
import com.example.client_trial2.Networking.Helpers.NetHelper;
import com.example.client_trial2.Networking.Interfaces.NetEventsInterface;
import com.example.client_trial2.Networking.Models.Packet;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.net.Socket;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

public class ClientSocket
{
    //region Private Fields
    //Synchronized Threading Fields
    ConcurrentLinkedQueue<byte[]> BufferQueue;
    ExecutorService Executor;
    //Client Fields
    private final NetConfig Config;
    private NetEventsInterface Listener;
    private Socket InnerSocket;
    private boolean EnableReconnect;
    //Message Fields
    private int MessageSize;
    //Buffering Fields
    private DataInputStream InComeStream;
    private DataOutputStream OutComeStream;
    private BufferState State;
    private AtomicBoolean IsBuffering;
    private byte[] Buffer;
    private byte[] Body;
    private byte[] Header;
    //Offset Fields
    private int ReadOffset;
    private int WriteOffset;
    private int BytesToProcess;
    //endregion

    //region Properties
    public boolean IsConnected;
    public PacketManager PacketManager;
    //endregion

    public ClientSocket(NetConfig Config)
    {
        this.Config = Config;
        InitializeClientSocket();
    }

    //region Private Methods
    private void InitializeClientSocket()
    {
        try
        {
            //Synchronized Threading Fields
            Executor = Executors.newCachedThreadPool();
            BufferQueue = new ConcurrentLinkedQueue<>();
            //Client Fields
            PacketManager = new PacketManager();
            IsConnected = false;
            EnableReconnect = true;
            //Message Fields
            MessageSize = -1;
            //Buffering Fields
            State = BufferState.HEADER;
            IsBuffering = new AtomicBoolean(false);
            Buffer = new byte[Config.GetBufferSize()];
            Header = new byte[Config.GetHeaderSize()];
            //Offset Fields
            ReadOffset = 0;
            WriteOffset = 0;
            BytesToProcess = 1024;
        }
        catch (Exception ex)
        {
            Listener.SetOnClientException(ex);
        }
    }

    private boolean Connect()
    {
        if (!IsConnected)
        {
            try
            {
                InnerSocket = new Socket(Config.GetHost(), Config.GetPort());
                OutComeStream = new DataOutputStream(InnerSocket.getOutputStream());
                InComeStream = new DataInputStream(InnerSocket.getInputStream());
                IsConnected = true;
                Listener.SetOnClientConnect(this);
            }
            catch (Exception ex)
            {
                IsConnected = false;
                Listener.SetOnClientException(ex);
                Disconnect();
            }
        }
        return IsConnected;
    }

    private void BeginReceive()
    {
        if (IsConnected)
        {
            new Thread(() ->
            {
                try
                {
                    while (true)
                    {
                        int BytesReceived = InComeStream.read(Buffer, 0, Buffer.length);
                        if (!ProcessReceive(BytesReceived))
                        {
                            break;
                        }
                    }
                }
                catch (Exception ex)
                {
                    Listener.SetOnClientException(ex);
                    Disconnect();
                }
            }).start();
        }
    }

    private boolean ProcessReceive(int BytesReceived)
    {
        if (BytesReceived > 0)
        {
            byte[] Packet = new byte[BytesReceived];
            System.arraycopy(Buffer, 0, Packet, 0, BytesReceived);
            Produce(Packet);
            return true;
        }
        else
        {
            Listener.SetOnClientException(new Exception("0 Bytes received."));
            Disconnect();
        }
        return false;
    }

    private void Produce(byte[] Packet)
    {
        try
        {
            BufferQueue.add(Packet);
            if (!IsBuffering.get())
            {
                IsBuffering.set(true);
                Executor.execute(this::HandleBuffering);
            }
        }
        catch (Exception ex)
        {
            Listener.SetOnClientException(ex);
            Disconnect();
        }
    }

    private void HandleBuffering()
    {
        while (true)
        {
            byte[] Packet;
            if (BufferQueue.size() == 0)
            {
                IsBuffering.set(false);
                break;
            }
            Packet = BufferQueue.poll();
            if (Packet != null)
            {
                BytesToProcess = Packet.length;
                while (BytesToProcess > 0)
                {
                    switch (State)
                    {
                        case HEADER:
                            if (BytesToProcess + WriteOffset >= Config.GetHeaderSize())
                            {
                                int ExactLength = (BytesToProcess >= Config.GetHeaderSize()) ? Config.GetHeaderSize() - WriteOffset : BytesToProcess;
                                System.arraycopy(Packet, ReadOffset, Header, WriteOffset, ExactLength);

                                WriteOffset = 0;
                                ReadOffset = ReadOffset + ExactLength;
                                BytesToProcess = BytesToProcess - ExactLength;
                                MessageSize = NetHelper.GetInt(Header);

                                if (MessageSize <= 0 || MessageSize >= Config.GetMaximumMessageSize())
                                {
                                    Listener.SetOnClientException(new Exception("Corrupted Header, " + MessageSize));
                                    BytesToProcess = 0;
                                    State = BufferState.NONE;
                                    Disconnect();
                                }
                                else
                                {
                                    State = BufferState.BODY;
                                }
                            }
                            else
                            {
                                System.arraycopy(Packet, ReadOffset, Header, WriteOffset, BytesToProcess);
                                WriteOffset = WriteOffset + BytesToProcess;
                                BytesToProcess = 0;
                            }
                            break;

                        case BODY:
                            if (Body == null)
                            {
                                Body = new byte[MessageSize];
                            }
                            else
                            {
                                if (Body.length != MessageSize)
                                {
                                    Body = new byte[MessageSize];
                                }
                            }

                            int BodyLength = (BytesToProcess + WriteOffset > MessageSize) ? MessageSize - WriteOffset : BytesToProcess;

                            System.arraycopy(Packet, ReadOffset, Body, WriteOffset, BodyLength);

                            WriteOffset = WriteOffset + BodyLength;
                            ReadOffset = ReadOffset + BodyLength;
                            BytesToProcess = BytesToProcess - BodyLength;

                            if (WriteOffset == MessageSize)
                            {
                                Listener.SetOnClientReceive(this, Body);
                                State = BufferState.HEADER;
                                WriteOffset = 0;

                            }
                            break;
                    }
                    if (BytesToProcess == 0)
                    {
                        ReadOffset = 0;
                    }
                }
            }
        }
    }
    //endregion

    //region Public Methods
    public void SetEventsListener(NetEventsInterface Listener) {
        this.Listener = Listener;
    }

    public void StartAttemptToConnect() {
        new Thread(() ->
        {
            while (this.EnableReconnect) {
                try {
                    if (!IsConnected) {
                        Listener.SetOnClientException(new Exception("Attempting to reconnect to server"));
                        if (Connect()) {
                            BeginReceive();
                        }
                    } else {
                        Thread.sleep(Config.GetReconnectInterval());
                    }
                } catch (Exception ex) {
                    Listener.SetOnClientException(ex);
                }
            }
        }).start();
    }

    public void StopAttemptToConnect() {
        if (EnableReconnect) {
            EnableReconnect = false;
        }
    }

    public void Disconnect() {
        if (IsConnected) {
            try {
                if (!InnerSocket.isInputShutdown()) {
                    InnerSocket.shutdownInput();
                }
                if (!InnerSocket.isOutputShutdown()) {
                    InnerSocket.shutdownOutput();
                }
                InnerSocket.close();
                IsConnected = false;
                Listener.SetOnClientDisconnect(this);
            } catch (Exception ex) {
                Listener.SetOnClientException(ex);
            }
        }
    }

    public void Send(Packet Message) {
        if (IsConnected)
        {
            try
            {
                byte[] PacketBytes = NetHelper.AppendHeader(Message.ToPacket(), Config.GetHeaderSize());
                OutComeStream.write(PacketBytes, 0, PacketBytes.length);
                OutComeStream.flush();
                Listener.SetOnClientSend(this, Message);
            }
            catch (Exception ex)
            {
                Listener.SetOnClientException(ex);
            }
        }
    }

    public String toString() {
        return Config.GetHost() + ":" + Config.GetPort();
    }
//endregion
}