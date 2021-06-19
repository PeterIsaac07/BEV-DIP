package com.example.client_trial2.Networking;

import android.util.Log;

import com.example.client_trial2.Networking.Models.Packet;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Hashtable;

public class PacketManager
{
    private final Hashtable<Short, Packet> Packets;

    public PacketManager()
    {
        Packets = new Hashtable<>();
    }

    public void AddPacket(Packet Packet)
    {
        if (!Packets.containsKey(Packet.GetCode()))
        {
            Packets.put(Packet.GetCode(), Packet);
        }
    }

    public void RemovePacket(Packet Packet)
    {
        Packets.remove(Packet.GetCode());
    }

    public boolean IsCodeValid(Short Code)
    {
        return Packets.containsKey(Code);
    }

    public String GetPacketName(Short Code)
    {
        Packet PackObj = Packets.get(Code);
        if (PackObj != null)
        {
            return PackObj.GetName();
        }
        return "N/A";
    }

    public byte[] GetPacketContent(Short Code)
    {
        if (Packets.containsKey(Code))
        {
            return Packets.get(Code).GetContent();
        }
        return null;
    }

    public Packet GetPacket(byte[] Bytes)
    {
        Packet Message;
        ByteBuffer wrapped = ByteBuffer.wrap(Bytes);
        wrapped.order(ByteOrder.LITTLE_ENDIAN);
        Short Code = wrapped.getShort();
        if (IsCodeValid(Code))
        {
            byte[] Content = new byte[Bytes.length - 2];
            System.arraycopy(Bytes, 2, Content, 0, Content.length);
            Message = new Packet(Code, GetPacketName(Code), Content);
        }
        else
        {
            Message = new Packet((short) -1, "", null);
        }
        return Message;
    }
}
