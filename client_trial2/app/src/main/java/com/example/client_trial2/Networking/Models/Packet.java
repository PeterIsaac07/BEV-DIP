package com.example.client_trial2.Networking.Models;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class Packet
{
    //region Constants
    private final int CODE_SIZE = 2;
    //endregion

    //region Private Fields
    private final short Code;
    private final String Name;
    private final byte[] Content;
    //endregion

    public Packet(short Code, String Name, byte[] Content)
    {
        this.Code = Code;
        this.Name = Name;
        this.Content = Content;
    }

    public short GetCode()
    {
        return Code;
    }

    public String GetName()
    {
        return Name;
    }

    public byte[] GetContent()
    {
        return Content;
    }

    public byte[] ToPacket()
    {
        byte[] Pack;
        if (Content != null)
        {
            Pack = new byte[CODE_SIZE + Content.length];
            byte[] CodeBytes = ByteBuffer.allocate(CODE_SIZE).order(ByteOrder.LITTLE_ENDIAN).putShort(Code).array();
            System.arraycopy(CodeBytes, 0, Pack, 0, CODE_SIZE);
            System.arraycopy(Content, 0, Pack, CODE_SIZE, Content.length);
        }
        else
        {
            Pack = new byte[CODE_SIZE];
            byte[] CodeBytes = ByteBuffer.allocate(CODE_SIZE).order(ByteOrder.LITTLE_ENDIAN).putShort(Code).array();
            System.arraycopy(CodeBytes, 0, Pack, 0, CODE_SIZE);
        }
        return Pack;
    }
}