package com.example.client_trial2.Networking;

public class NetConfig
{
    private final String Host;
    private final int Port;
    private final int BufferSize;
    private final int HeaderSize;
    private final int MaximumMessageSize;
    private final int ReconnectInterval;

    public NetConfig()
    {
        Host = "192.168.1.227";
        Port = 65432;
        BufferSize = 1024 * 1024*10;
        HeaderSize = 4;
        MaximumMessageSize = 1024 * 1024 * 5 ;
        ReconnectInterval = 250;
    }

    public NetConfig(String Host, int Port, int BufferSize, int HeaderSize, int MaximumMessageSize, int ReconnectInterval)
    {
        this.Host = Host;
        this.Port = Port;
        this.BufferSize = BufferSize;
        this.HeaderSize = HeaderSize;
        this.MaximumMessageSize = MaximumMessageSize;
        this.ReconnectInterval = ReconnectInterval;
    }

    public String GetHost()
    {
        return Host;
    }

    public int GetPort()
    {
        return Port;
    }

    public int GetBufferSize()
    {
        return BufferSize;
    }

    public int GetHeaderSize()
    {
        return HeaderSize;
    }

    public int GetMaximumMessageSize()
    {
        return MaximumMessageSize;
    }

    public int GetReconnectInterval()
    {
        return ReconnectInterval;
    }
}